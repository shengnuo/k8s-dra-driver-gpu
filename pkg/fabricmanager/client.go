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

// Package fabricmanager talks to the NVIDIA Fabric Manager (FM) daemon to
// discover the static GPU partition layout used on NVSwitch-based HGX systems
// and exposes lookup APIs so the DRA driver can map a PCI bus ID to the
// gpuModuleId it should advertise and to the FM partitions the GPU may
// participate in.
//
// Fabric Manager exposes a C SDK (libnvidia-fabricmanager.so) with calls like
// fmLibInit / fmConnect / fmGetSupportedFabricPartitions described in the FM
// User Guide. The Client interface in this package mirrors that C API one
// method at a time so a real cgo-backed implementation (or NVIDIA's
// go-fabric-manager bindings) can be plugged in without changing any caller.
package fabricmanager

import "errors"

// ErrUnimplemented is returned by the stub client. Callers can use
// errors.Is(err, ErrUnimplemented) to detect that no real FM backend is wired
// up and degrade gracefully (e.g. skip publishing FM-derived attributes).
var ErrUnimplemented = errors.New("fabricmanager: client backend not implemented")

// ConnectParams matches fmConnectParams_t. AddressInfo is the host:port (or
// unix socket path) that nv-fabricmanager is listening on. The FM daemon's
// default TCP port is 6666; for in-pod deployments a unix socket is typical.
type ConnectParams struct {
	// AddressInfo is "<host>:<port>" for TCP, or a filesystem path when
	// AddressIsUnixSocket is true. Empty means use the FM SDK default.
	AddressInfo string
	// TimeoutMs is the connect timeout, in milliseconds. Zero uses the SDK
	// default.
	TimeoutMs uint32
	// AddressIsUnixSocket selects the unix-socket transport.
	AddressIsUnixSocket bool
}

// PartitionGPU describes a single GPU as reported by FM inside a fabric
// partition. The fields correspond to fmFabricPartitionGpuInfo_t.
//
// PhysicalID is the GPU's physical/module ID; it is the same value that
// nvmlDeviceGetModuleId returns and that the design doc calls gpuModuleId.
type PartitionGPU struct {
	PhysicalID          int
	UUID                string
	PCIBusID            string
	NumNvLinksAvailable uint32
	MaxNumNvLinks       uint32
	NvLinkLineRateMBps  uint32
}

// Partition is a single FM-supported fabric partition: an ordered set of
// GPUs that may be allocated together with a guaranteed NVLink topology.
// Corresponds to fmFabricPartitionInfo_t.
type Partition struct {
	ID       int
	IsActive bool
	GPUs     []PartitionGPU
}

// GPUModuleIDs is a convenience accessor that returns the PhysicalIDs of all
// GPUs in the partition, in the order FM reported them. This matches the
// "gpuModuleIds" field in the design doc's static partition JSON.
func (p Partition) GPUModuleIDs() []int {
	ids := make([]int, len(p.GPUs))
	for i, g := range p.GPUs {
		ids[i] = g.PhysicalID
	}
	return ids
}

// PCIDevice identifies a PCI(e) function. It mirrors fmPciDevice_t and is
// used to describe SR-IOV virtual functions when activating a partition for a
// vGPU workload.
type PCIDevice struct {
	Domain   uint32
	Bus      uint32
	Device   uint32
	Function uint32
}

// UnsupportedPartition describes an FM-rejected partition (one that exists
// topologically but cannot be activated, e.g. due to NVLink failures).
// Mirrors fmUnsupportedFabricPartitionInfo_t.
type UnsupportedPartition struct {
	ID             int
	GPUPhysicalIDs []int
}

// NvlinkFailedDevice describes a GPU or NVSwitch with one or more NVLink
// ports that have failed training. Mirrors fmNvlinkFailedDeviceInfo_t.
//
// On HGX-H100/H200 and later (ALI-trained NVLinks), FM does not surface
// failures through this API and always returns an empty list.
type NvlinkFailedDevice struct {
	UUID     string
	PCIBusID string
	Ports    []uint32
}

// NvlinkFailedDevices is the union of failed GPUs and failed NVSwitches, as
// reported by fmGetNvlinkFailedDevices.
type NvlinkFailedDevices struct {
	GPUs     []NvlinkFailedDevice
	Switches []NvlinkFailedDevice
}

// Client is a Go projection of the NVIDIA Fabric Manager C SDK
// (libnvidia-fabricmanager.so). The methods are intentionally 1:1 with the
// underlying C calls (fmLibInit, fmConnect, fmGetSupportedFabricPartitions,
// fmActivateFabricPartition, ...) so a cgo-backed implementation (or
// NVIDIA's go-fabric-manager package, once available) can satisfy this
// interface without any translation layer.
//
// Typical lifecycle:
//
//	c.Init()                                   // load FM library
//	c.Connect(params)                          // connect to nv-fabricmanager
//	c.GetSupportedFabricPartitions()           // discovery
//	c.ActivateFabricPartition(id)              // on DRA claim allocate
//	c.DeactivateFabricPartition(id)            // on DRA claim release
//	c.Disconnect()
//	c.Shutdown()
//
// All methods may be called concurrently as long as the underlying FM SDK
// supports it; the stub backend is trivially safe.
type Client interface {
	// Init loads/initializes the FM library (fmLibInit).
	Init() error

	// Connect establishes a connection to the nv-fabricmanager daemon
	// (fmConnect). It must be called after Init and before any partition
	// query/activation call.
	Connect(params ConnectParams) error

	// GetSupportedFabricPartitions returns every partition FM supports on
	// this node, including each partition's GPU members. This is the call
	// that produces the data the design doc's static-partition JSON used to
	// describe.
	GetSupportedFabricPartitions() ([]Partition, error)

	// GetUnsupportedFabricPartitions returns partitions FM knows about but
	// has marked unsupported (e.g. due to NVLink failures). On HGX-H100 and
	// later, this list is always empty.
	// Mirrors fmGetUnsupportedFabricPartitions.
	GetUnsupportedFabricPartitions() ([]UnsupportedPartition, error)

	// ActivateFabricPartition asks FM to program the NVSwitch fabric for the
	// given partition. Used as part of DRA allocation for GPU passthrough.
	// Mirrors fmActivateFabricPartition.
	//
	// Returns nil on success. The FM SDK reports FM_ST_IN_USE when the
	// partition (or one of its GPUs) is already attached to another tenant;
	// callers should treat that as an allocation conflict, not a fatal
	// error.
	ActivateFabricPartition(partitionID int) error

	// ActivateFabricPartitionWithVFs activates a partition that contains
	// SR-IOV virtual functions, used for vGPU workloads. The vfList must
	// have one entry per GPU in the partition, in the same order as
	// reported by GetSupportedFabricPartitions.
	// Mirrors fmActivateFabricPartitionWithVFs.
	ActivateFabricPartitionWithVFs(partitionID int, vfList []PCIDevice) error

	// DeactivateFabricPartition releases the NVSwitch fabric programming
	// for the given partition. Used when a DRA claim is released.
	// Mirrors fmDeactivateFabricPartition.
	DeactivateFabricPartition(partitionID int) error

	// SetActivatedFabricPartitions tells FM the authoritative list of
	// currently-activated partitions. This is the FM resiliency hook
	// (FABRIC_MODE_RESTART=1): after FM restarts, the DRA driver replays
	// the partitions it had previously activated so FM rebuilds its
	// internal state without disrupting running VMs.
	// Mirrors fmSetActivatedFabricPartitions.
	SetActivatedFabricPartitions(partitionIDs []int) error

	// GetNvlinkFailedDevices returns GPUs and NVSwitches with failed
	// NVLinks. Useful for surfacing fabric-allocation visibility (REQ-4 in
	// the design doc). On HGX-H100 and later, this list is always empty
	// because NVLinks are trained autonomously (ALI).
	// Mirrors fmGetNvlinkFailedDevices.
	GetNvlinkFailedDevices() (NvlinkFailedDevices, error)

	// Disconnect closes the connection to nv-fabricmanager (fmDisconnect).
	Disconnect() error

	// Shutdown unloads/shuts down the FM library (fmLibShutdown).
	Shutdown() error
}
