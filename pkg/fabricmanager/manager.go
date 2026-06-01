/*
 * Copyright (c) 2026 NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package fabricmanager

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// Manager is the in-memory view this package exposes to the rest of the DRA
// driver. It owns:
//
//   - a PCI bus ID <-> gpuModuleId map populated by walking every visible GPU
//     via NVML's DeviceGetModuleId (per the design doc, this is the value
//     advertised on the ResourceSlice as `gpuModuleId`);
//   - the list of FM-supported fabric partitions discovered through a Client
//     (i.e. through nv-fabricmanager itself, not a static JSON file);
//   - a long-lived connection to nv-fabricmanager so allocation-time
//     operations (Activate/Deactivate) don't have to re-Init/Connect on every
//     call.
//
// Lookups across NVML and FM data are joined on gpuModuleId, since FM
// reports each member GPU's physicalId which is the same value
// nvmlDeviceGetModuleId returns.
//
// A Manager is created via Open and must be closed via Close. Manager methods
// are safe for concurrent use.
type Manager struct {
	mu sync.RWMutex

	client Client
	opened bool

	moduleIDByPCI map[string]int
	pciByModuleID map[int]string

	partitionsByID map[int]Partition

	// activated tracks the set of partitions Activate has been called on
	// (and Deactivate has not yet undone) so the DRA driver can replay it
	// to FM after a daemon restart via SyncActivatedPartitionsToFM.
	activated map[int]struct{}
}

// NVMLDeviceLister is the minimal subset of nvml.Interface required to build
// the gpuModuleId <-> PCI bus ID mapping. Using a narrow interface keeps the
// Manager testable without standing up a full NVML library; production code
// can pass a real nvml.Interface, which already satisfies it.
type NVMLDeviceLister interface {
	DeviceGetCount() (int, nvml.Return)
	DeviceGetHandleByIndex(int) (nvml.Device, nvml.Return)
}

// Open builds a Manager and leaves a long-lived FM connection in place so the
// caller can subsequently activate and deactivate partitions. It:
//
//  1. Walks every GPU visible to NVML and records (PCI bus ID, gpuModuleId).
//  2. Calls Init+Connect on the FM client.
//  3. Calls GetSupportedFabricPartitions and sanity-checks every
//     gpuModuleId FM mentions against NVML.
//
// On any failure Open cleans up whatever it had set up and returns the
// error; on success the caller is responsible for calling Close.
//
// NVML must already have been initialized by the caller; this package
// follows the same convention as cmd/gpu-kubelet-plugin/nvlib.go and does
// not own the NVML lifetime.
func Open(lib NVMLDeviceLister, client Client, params ConnectParams) (*Manager, error) {
	if lib == nil {
		return nil, fmt.Errorf("fabricmanager: nil NVML interface")
	}
	if client == nil {
		return nil, fmt.Errorf("fabricmanager: nil FM client")
	}

	m := &Manager{
		client:         client,
		moduleIDByPCI:  make(map[string]int),
		pciByModuleID:  make(map[int]string),
		partitionsByID: make(map[int]Partition),
		activated:      make(map[int]struct{}),
	}

	if err := m.refreshFromNVML(lib); err != nil {
		return nil, err
	}

	if err := client.Init(); err != nil {
		return nil, fmt.Errorf("fabricmanager: fmLibInit: %w", err)
	}
	if err := client.Connect(params); err != nil {
		_ = client.Shutdown()
		return nil, fmt.Errorf("fabricmanager: fmConnect(%q): %w", params.AddressInfo, err)
	}

	partitions, err := client.GetSupportedFabricPartitions()
	if err != nil {
		_ = client.Disconnect()
		_ = client.Shutdown()
		return nil, fmt.Errorf("fabricmanager: fmGetSupportedFabricPartitions: %w", err)
	}
	if err := m.installPartitions(partitions); err != nil {
		_ = client.Disconnect()
		_ = client.Shutdown()
		return nil, err
	}

	// Seed activated set from any partitions FM already reports as active
	// (e.g. after a DRA driver restart with FM still running).
	for _, p := range partitions {
		if p.IsActive {
			m.activated[p.ID] = struct{}{}
		}
	}

	m.opened = true
	return m, nil
}

// Close tears down the Manager's FM connection. Calling Close on an
// already-closed Manager is a no-op. Errors from Disconnect and Shutdown are
// joined; the second error does not mask the first.
func (m *Manager) Close() error {
	m.mu.Lock()
	if !m.opened {
		m.mu.Unlock()
		return nil
	}
	m.opened = false
	client := m.client
	m.mu.Unlock()

	var firstErr error
	if err := client.Disconnect(); err != nil {
		firstErr = fmt.Errorf("fabricmanager: fmDisconnect: %w", err)
	}
	if err := client.Shutdown(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("fabricmanager: fmLibShutdown: %w", err)
	}
	return firstErr
}

// refreshFromNVML populates the gpuModuleId <-> PCI bus ID maps by walking
// every NVML-visible GPU. The maps are replaced atomically so concurrent
// readers always see a consistent snapshot.
func (m *Manager) refreshFromNVML(lib NVMLDeviceLister) error {
	count, ret := lib.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("fabricmanager: NVML DeviceGetCount: %v", ret)
	}

	moduleIDByPCI := make(map[string]int, count)
	pciByModuleID := make(map[int]string, count)

	for i := 0; i < count; i++ {
		dev, ret := lib.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("fabricmanager: NVML DeviceGetHandleByIndex(%d): %v", i, ret)
		}

		moduleID, ret := dev.GetModuleId()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("fabricmanager: NVML GetModuleId for device %d: %v", i, ret)
		}

		pciInfo, ret := dev.GetPciInfo()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("fabricmanager: NVML GetPciInfo for device %d: %v", i, ret)
		}
		pciBusID := normalizePCIBusID(cString(pciInfo.BusId[:]))
		if pciBusID == "" {
			return fmt.Errorf("fabricmanager: empty PCI bus ID for device %d (moduleId=%d)", i, moduleID)
		}

		if existing, ok := pciByModuleID[moduleID]; ok {
			return fmt.Errorf("fabricmanager: duplicate gpuModuleId %d for PCI bus IDs %q and %q",
				moduleID, existing, pciBusID)
		}
		if existing, ok := moduleIDByPCI[pciBusID]; ok {
			return fmt.Errorf("fabricmanager: duplicate PCI bus ID %q for gpuModuleIds %d and %d",
				pciBusID, existing, moduleID)
		}
		moduleIDByPCI[pciBusID] = moduleID
		pciByModuleID[moduleID] = pciBusID
	}

	m.mu.Lock()
	m.moduleIDByPCI = moduleIDByPCI
	m.pciByModuleID = pciByModuleID
	m.mu.Unlock()
	return nil
}

// installPartitions records the FM-supplied partitions, validating that every
// gpuModuleId referenced by FM corresponds to a GPU NVML enumerated on this
// node. A mismatch usually indicates that NVML and FM disagree about the
// node's GPU inventory and is treated as an error.
func (m *Manager) installPartitions(parts []Partition) error {
	byID := make(map[int]Partition, len(parts))

	m.mu.RLock()
	pciByModuleID := m.pciByModuleID
	m.mu.RUnlock()

	for _, p := range parts {
		if _, dup := byID[p.ID]; dup {
			return fmt.Errorf("fabricmanager: FM returned duplicate partitionId %d", p.ID)
		}
		if len(p.GPUs) == 0 {
			return fmt.Errorf("fabricmanager: partition %d has no GPUs", p.ID)
		}
		seen := make(map[int]struct{}, len(p.GPUs))
		for _, g := range p.GPUs {
			if _, dup := seen[g.PhysicalID]; dup {
				return fmt.Errorf("fabricmanager: partition %d references gpuModuleId %d twice",
					p.ID, g.PhysicalID)
			}
			seen[g.PhysicalID] = struct{}{}
			if _, known := pciByModuleID[g.PhysicalID]; !known {
				return fmt.Errorf("fabricmanager: partition %d references unknown gpuModuleId %d (not present in NVML)",
					p.ID, g.PhysicalID)
			}
		}
		byID[p.ID] = p
	}

	m.mu.Lock()
	m.partitionsByID = byID
	m.mu.Unlock()
	return nil
}

// GetModuleIDByPCI returns the gpuModuleId associated with the given PCI bus
// ID, or false if no GPU on the node matches. Lookup is case-insensitive and
// tolerates both the 4-digit ("0000:3b:00.0") and 8-digit ("00000000:3B:00.0")
// PCI domain forms.
func (m *Manager) GetModuleIDByPCI(pciBusID string) (int, bool) {
	key := normalizePCIBusID(pciBusID)
	m.mu.RLock()
	defer m.mu.RUnlock()
	id, ok := m.moduleIDByPCI[key]
	return id, ok
}

// GetPCIByModuleID returns the PCI bus ID associated with the given
// gpuModuleId, or false if no GPU on the node matches.
func (m *Manager) GetPCIByModuleID(moduleID int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	pci, ok := m.pciByModuleID[moduleID]
	return pci, ok
}

// GetPartition returns the FM partition info for the given partitionId, or
// false if no such partition was reported by FM.
func (m *Manager) GetPartition(partitionID int) (Partition, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.partitionsByID[partitionID]
	return p, ok
}

// Partitions returns every FM-supported partition, sorted ascending by id.
func (m *Manager) Partitions() []Partition {
	m.mu.RLock()
	ids := make([]int, 0, len(m.partitionsByID))
	for id := range m.partitionsByID {
		ids = append(ids, id)
	}
	out := make([]Partition, 0, len(ids))
	sort.Ints(ids)
	for _, id := range ids {
		out = append(out, m.partitionsByID[id])
	}
	m.mu.RUnlock()
	return out
}

// GetPartitionsByPCI returns all partitionIds that include the GPU identified
// by the given PCI bus ID. Sorted ascending. Returns (nil, false) if the PCI
// bus ID is unknown to NVML.
func (m *Manager) GetPartitionsByPCI(pciBusID string) ([]int, bool) {
	moduleID, ok := m.GetModuleIDByPCI(pciBusID)
	if !ok {
		return nil, false
	}
	return m.GetPartitionsByModuleID(moduleID), true
}

// GetPartitionsByModuleID returns all partitionIds that include the given
// gpuModuleId. Sorted ascending. Returns an empty (non-nil) slice if no
// partitions reference the module.
func (m *Manager) GetPartitionsByModuleID(moduleID int) []int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var ids []int
	for _, p := range m.partitionsByID {
		for _, g := range p.GPUs {
			if g.PhysicalID == moduleID {
				ids = append(ids, p.ID)
				break
			}
		}
	}
	sort.Ints(ids)
	if ids == nil {
		ids = []int{}
	}
	return ids
}

// GetPartitionsBySizeByModuleID returns a map keyed by partition size (number
// of GPUs in the partition) to the partitionId of the partition of that size
// that includes the given gpuModuleId. This is the shape the design doc's
// "Fabric Manager Advertised by Partition" strategy publishes on each
// ResourceSlice device, e.g.:
//
//	gpuModuleId: 1
//	partition1:  8
//	partition2:  4
//	partition4:  2
//	partition8:  1
//
// On a well-formed HGX node FM produces exactly one partition per
// (size, GPU) pair; if more than one is found this method returns an error
// to surface the topology inconsistency rather than silently dropping data.
func (m *Manager) GetPartitionsBySizeByModuleID(moduleID int) (map[int]int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make(map[int]int)
	for _, p := range m.partitionsByID {
		size := len(p.GPUs)
		for _, g := range p.GPUs {
			if g.PhysicalID == moduleID {
				if existing, dup := out[size]; dup {
					return nil, fmt.Errorf(
						"fabricmanager: gpuModuleId %d appears in two partitions of size %d (%d and %d)",
						moduleID, size, existing, p.ID)
				}
				out[size] = p.ID
				break
			}
		}
	}
	return out, nil
}

// GetPartitionsBySizeByPCI is a convenience wrapper around
// GetPartitionsBySizeByModuleID that accepts a PCI bus ID. Returns
// (nil, false, nil) if the PCI bus ID is unknown to NVML; (m, true, nil) on
// success; or (nil, true, err) if the partition layout is inconsistent.
func (m *Manager) GetPartitionsBySizeByPCI(pciBusID string) (map[int]int, bool, error) {
	moduleID, ok := m.GetModuleIDByPCI(pciBusID)
	if !ok {
		return nil, false, nil
	}
	out, err := m.GetPartitionsBySizeByModuleID(moduleID)
	return out, true, err
}

// ActivatePartition asks Fabric Manager to program the NVSwitch fabric for
// the given partition. This is the call DRA allocation should issue once it
// has selected a fabric-compatible GPU set (per the design doc, §4.1 step 3).
//
// On success the partition is recorded in the Manager's activated set so it
// can be replayed to FM after a daemon restart via SyncActivatedPartitionsToFM.
func (m *Manager) ActivatePartition(partitionID int) error {
	if err := m.checkOpenPartition(partitionID); err != nil {
		return err
	}
	if err := m.client.ActivateFabricPartition(partitionID); err != nil {
		return fmt.Errorf("fabricmanager: fmActivateFabricPartition(%d): %w", partitionID, err)
	}
	m.markActivated(partitionID, true)
	return nil
}

// ActivatePartitionWithVFs activates a partition and binds SR-IOV virtual
// functions to its GPUs, used for vGPU workloads (REQ-8/9 in the design doc).
// The vfList must have one entry per GPU in the partition, in the same order
// the FM client returned them.
func (m *Manager) ActivatePartitionWithVFs(partitionID int, vfList []PCIDevice) error {
	if err := m.checkOpenPartition(partitionID); err != nil {
		return err
	}
	p, _ := m.GetPartition(partitionID)
	if len(vfList) != len(p.GPUs) {
		return fmt.Errorf("fabricmanager: partition %d expects %d VFs, got %d",
			partitionID, len(p.GPUs), len(vfList))
	}
	if err := m.client.ActivateFabricPartitionWithVFs(partitionID, vfList); err != nil {
		return fmt.Errorf("fabricmanager: fmActivateFabricPartitionWithVFs(%d): %w", partitionID, err)
	}
	m.markActivated(partitionID, true)
	return nil
}

// DeactivatePartition releases an activated partition. It should be called
// when the corresponding DRA claim is released (design doc §4.1 step 5).
func (m *Manager) DeactivatePartition(partitionID int) error {
	if err := m.checkOpenPartition(partitionID); err != nil {
		return err
	}
	if err := m.client.DeactivateFabricPartition(partitionID); err != nil {
		return fmt.Errorf("fabricmanager: fmDeactivateFabricPartition(%d): %w", partitionID, err)
	}
	m.markActivated(partitionID, false)
	return nil
}

// ActivatedPartitions returns the set of partition IDs the Manager has
// observed activated (either reported active by FM at Open time or activated
// by this Manager since). Sorted ascending.
func (m *Manager) ActivatedPartitions() []int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	ids := make([]int, 0, len(m.activated))
	for id := range m.activated {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

// SyncActivatedPartitionsToFM tells FM the authoritative list of currently
// activated partitions. Used after FM restarts in resiliency mode
// (FABRIC_MODE_RESTART=1) to rebuild FM's internal state without disturbing
// running VMs.
func (m *Manager) SyncActivatedPartitionsToFM() error {
	if err := m.checkOpen(); err != nil {
		return err
	}
	ids := m.ActivatedPartitions()
	if err := m.client.SetActivatedFabricPartitions(ids); err != nil {
		return fmt.Errorf("fabricmanager: fmSetActivatedFabricPartitions(%v): %w", ids, err)
	}
	return nil
}

// RefreshPartitions re-fetches the supported partition list from FM and
// re-runs the NVML cross-check. Useful if FM topology can change at runtime
// (rare on HGX, but supported by the SDK). The activated set is preserved
// for partitions that still exist; partitions that disappeared are dropped.
func (m *Manager) RefreshPartitions() error {
	if err := m.checkOpen(); err != nil {
		return err
	}
	parts, err := m.client.GetSupportedFabricPartitions()
	if err != nil {
		return fmt.Errorf("fabricmanager: fmGetSupportedFabricPartitions: %w", err)
	}
	if err := m.installPartitions(parts); err != nil {
		return err
	}
	m.mu.Lock()
	for id := range m.activated {
		if _, ok := m.partitionsByID[id]; !ok {
			delete(m.activated, id)
		}
	}
	for _, p := range parts {
		if p.IsActive {
			m.activated[p.ID] = struct{}{}
		}
	}
	m.mu.Unlock()
	return nil
}

// UnsupportedPartitions returns partitions FM has marked unsupported (e.g.
// because of NVLink failures). On HGX-H100 and later this is always empty.
func (m *Manager) UnsupportedPartitions() ([]UnsupportedPartition, error) {
	if err := m.checkOpen(); err != nil {
		return nil, err
	}
	parts, err := m.client.GetUnsupportedFabricPartitions()
	if err != nil {
		return nil, fmt.Errorf("fabricmanager: fmGetUnsupportedFabricPartitions: %w", err)
	}
	return parts, nil
}

// NvlinkFailedDevices returns GPUs and NVSwitches with failed NVLinks.
// On HGX-H100 and later this is always empty (NVLinks are autonomously
// trained via ALI). Surfacing this gives the DRA driver visibility into
// fabric failures (REQ-4) on older HGX generations.
func (m *Manager) NvlinkFailedDevices() (NvlinkFailedDevices, error) {
	if err := m.checkOpen(); err != nil {
		return NvlinkFailedDevices{}, err
	}
	failed, err := m.client.GetNvlinkFailedDevices()
	if err != nil {
		return NvlinkFailedDevices{}, fmt.Errorf("fabricmanager: fmGetNvlinkFailedDevices: %w", err)
	}
	return failed, nil
}

func (m *Manager) checkOpen() error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if !m.opened {
		return fmt.Errorf("fabricmanager: manager is closed")
	}
	return nil
}

func (m *Manager) checkOpenPartition(partitionID int) error {
	if err := m.checkOpen(); err != nil {
		return err
	}
	if _, ok := m.GetPartition(partitionID); !ok {
		return fmt.Errorf("fabricmanager: unknown partitionId %d", partitionID)
	}
	return nil
}

func (m *Manager) markActivated(partitionID int, active bool) {
	m.mu.Lock()
	if active {
		m.activated[partitionID] = struct{}{}
	} else {
		delete(m.activated, partitionID)
	}
	if p, ok := m.partitionsByID[partitionID]; ok {
		p.IsActive = active
		m.partitionsByID[partitionID] = p
	}
	m.mu.Unlock()
}

// normalizePCIBusID converts a PCI bus ID into a canonical form that compares
// equal across producers. Producers differ in two ways:
//   - hex casing: NVML emits upper-case; sysfs / Kubernetes resource
//     attributes typically use lower-case.
//   - domain width: NVML's PciInfo.BusId uses an 8-digit domain
//     ("00000000:3B:00.0"), while sysfs and most Kubernetes attributes use a
//     4-digit domain ("0000:3b:00.0").
//
// We canonicalize to upper-case with an 8-digit domain. Inputs without a
// domain segment ("3b:00.0") or otherwise unparseable are returned upper-cased
// and trimmed without further mangling so callers still see a stable key.
func normalizePCIBusID(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		return s
	}
	domain, rest := parts[0], parts[1]
	switch {
	case len(domain) < 8:
		domain = strings.Repeat("0", 8-len(domain)) + domain
	case len(domain) > 8:
		domain = domain[len(domain)-8:]
	}
	return domain + ":" + rest
}

// cString converts a NUL-terminated byte slice (as used in NVML C structs)
// into a Go string by truncating at the first NUL.
func cString(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
