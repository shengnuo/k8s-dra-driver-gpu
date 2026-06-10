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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const DefaultFMPMBinary = "fmpm"

// defaultFMPMTimeout bounds how long any single fmpm invocation may run before
// the process is killed. fmpm performs its own (short) FM connect timeout
// internally
const defaultFMPMTimeout = 10 * time.Second

// fmpmClient is a Client backed by the NVIDIA "fmpm" command-line tool
// (https://github.com/NVIDIA/Fabric-Manager-Client).
//
// This backend is useful where building/loading libnvidia-fabricmanager.so in
// the driver process is undesirable: the only runtime dependency is the fmpm
// executable (which links the FM SDK itself). Each fmpm invocation performs a
// full fmLibInit/fmConnect/.../fmDisconnect/fmLibShutdown cycle, so this
// package's Init/Connect/Disconnect/Shutdown methods do not hold any live FM
// handle; Connect simply records the connection parameters that are then
// passed as flags to each fmpm call.
//
// Limitations relative to the cgo backend:
//   - ActivateFabricPartitionWithVFs is unsupported (the fmpm CLI exposes no
//     VF-activation option) and returns an error.
//   - ConnectParams.TimeoutMs is not forwarded; fmpm uses its own internal FM
//     connect timeout.
type fmpmClient struct {
	binaryPath string
	timeout    time.Duration

	params    ConnectParams
	connected bool

	// runner executes the fmpm binary with the given arguments and returns
	// its stdout. It is a field so tests can substitute a fake without a
	// real fmpm binary. A non-nil error for a non-zero exit is a
	// *fmpmExitError carrying the FM status (exit) code and captured stderr.
	runner func(args []string) ([]byte, error)
}

// NewFMPMClient returns a Client that drives Fabric Manager through the fmpm
// command-line tool. binaryPath optionally points at a specific fmpm
// executable; an empty string uses DefaultFMPMBinary resolved against $PATH.
//
// As with NewClient, constructing the client does not touch FM or the
// filesystem; the binary is resolved lazily by Init.
func NewFMPMClient(binaryPath string) Client {
	if binaryPath == "" {
		binaryPath = DefaultFMPMBinary
	}
	c := &fmpmClient{
		binaryPath: binaryPath,
		timeout:    defaultFMPMTimeout,
	}
	c.runner = c.execRunner
	return c
}

func (c *fmpmClient) Init() error {
	// Resolve the binary up front so a missing fmpm fails here (mirroring
	// fmLibInit failing when the FM library is absent) rather than on the
	// first partition query.
	path, err := exec.LookPath(c.binaryPath)
	if err != nil {
		return fmt.Errorf("fabricmanager: locating fmpm binary %q: %w", c.binaryPath, err)
	}
	c.binaryPath = path
	return nil
}

func (c *fmpmClient) Connect(params ConnectParams) error {
	c.params = params
	c.connected = true
	return nil
}

func (c *fmpmClient) Disconnect() error {
	c.connected = false
	return nil
}

func (c *fmpmClient) Shutdown() error {
	return nil
}

func (c *fmpmClient) GetSupportedFabricPartitions() ([]Partition, error) {
	if !c.connected {
		return nil, errNotConnected
	}
	out, err := c.run("-l")
	if err != nil {
		return nil, err
	}
	var list fmpmPartitionList
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("fabricmanager: parsing fmpm -l output: %w", err)
	}
	return list.toPartitions(), nil
}

func (c *fmpmClient) GetUnsupportedFabricPartitions() ([]UnsupportedPartition, error) {
	if !c.connected {
		return nil, errNotConnected
	}
	out, err := c.run("--list-unsupported-partitions")
	if err != nil {
		return nil, err
	}
	var list fmpmUnsupportedPartitionList
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("fabricmanager: parsing fmpm --list-unsupported-partitions output: %w", err)
	}
	return list.toUnsupportedPartitions(), nil
}

func (c *fmpmClient) ActivateFabricPartition(partitionID int) error {
	if !c.connected {
		return errNotConnected
	}
	_, err := c.run("-a", strconv.Itoa(partitionID))
	return err
}

// ActivateFabricPartitionWithVFs is not supported by the fmpm backend: the
// fmpm command-line tool exposes no option to bind SR-IOV virtual functions
// when activating a partition. Use the cgo-backed client (NewClient) for vGPU
// workloads.
func (c *fmpmClient) ActivateFabricPartitionWithVFs(partitionID int, vfList []PCIDevice) error {
	return fmt.Errorf("fabricmanager: fmpm backend does not support ActivateFabricPartitionWithVFs (partition %d)", partitionID)
}

func (c *fmpmClient) DeactivateFabricPartition(partitionID int) error {
	if !c.connected {
		return errNotConnected
	}
	_, err := c.run("-d", strconv.Itoa(partitionID))
	return err
}

func (c *fmpmClient) SetActivatedFabricPartitions(partitionIDs []int) error {
	if !c.connected {
		return errNotConnected
	}
	ids := make([]string, len(partitionIDs))
	for i, id := range partitionIDs {
		ids[i] = strconv.Itoa(id)
	}
	// fmpm expects a comma-separated list with no spaces.
	_, err := c.run("--set-activated-list", strings.Join(ids, ","))
	return err
}

func (c *fmpmClient) GetNvlinkFailedDevices() (NvlinkFailedDevices, error) {
	if !c.connected {
		return NvlinkFailedDevices{}, errNotConnected
	}
	out, err := c.run("--get-nvlink-failed-devices")
	if err != nil {
		return NvlinkFailedDevices{}, err
	}
	var devices fmpmNvlinkFailedDevices
	if err := json.Unmarshal(out, &devices); err != nil {
		return NvlinkFailedDevices{}, fmt.Errorf("fabricmanager: parsing fmpm --get-nvlink-failed-devices output: %w", err)
	}
	return devices.toNvlinkFailedDevices(), nil
}

// run prepends the connection flags derived from the stored ConnectParams and
// invokes the fmpm binary through the (possibly faked) runner.
func (c *fmpmClient) run(args ...string) ([]byte, error) {
	return c.runner(append(c.connectArgs(), args...))
}

// connectArgs translates the stored ConnectParams into fmpm flags. An empty
// AddressInfo lets fmpm use its own default (127.0.0.1 over TCP).
func (c *fmpmClient) connectArgs() []string {
	if c.params.AddressInfo == "" {
		return nil
	}
	if c.params.AddressIsUnixSocket {
		return []string{"--unix-domain-socket", c.params.AddressInfo}
	}
	return []string{"--hostname", c.params.AddressInfo}
}

// execRunner is the default runner: it executes the fmpm binary, returning its
// stdout. On a non-zero exit it returns a *fmpmExitError so callers can inspect
// the FM status code that fmpm propagates as its exit code.
func (c *fmpmClient) execRunner(args []string) ([]byte, error) {
	ctx := context.Background()
	if c.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, c.binaryPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return stdout.Bytes(), &fmpmExitError{
				Args:   args,
				Code:   exitErr.ExitCode(),
				Stdout: strings.TrimSpace(stdout.String()),
				Stderr: strings.TrimSpace(stderr.String()),
			}
		}
		return stdout.Bytes(), fmt.Errorf("fabricmanager: running %q %s: %w",
			c.binaryPath, strings.Join(args, " "), err)
	}
	return stdout.Bytes(), nil
}

// fmpmExitError reports a non-zero exit from the fmpm binary. fmpm returns the
// underlying fmReturn_t value as its process exit code, so Code is the FM
// status (e.g. FM_ST_IN_USE for an allocation conflict).
type fmpmExitError struct {
	Args   []string
	Code   int
	Stdout string
	Stderr string
}

func (e *fmpmExitError) Error() string {
	msg := e.Stderr
	if msg == "" {
		msg = e.Stdout
	}
	if msg == "" {
		msg = "(no output)"
	}
	return fmt.Sprintf("fabricmanager: fmpm %s failed (fmReturn=%d): %s",
		strings.Join(e.Args, " "), e.Code, msg)
}

// JSON types mirroring the documents that fmpm prints on stdout. Field names
// match the keys emitted by fmpm.cpp (which serializes the FM SDK structs via
// jsoncpp). Counts are carried for completeness but the conversions iterate the
// arrays directly, so a count/array mismatch cannot cause an out-of-range read.

type fmpmPartitionList struct {
	NumPartitions    uint32              `json:"numPartitions"`
	MaxNumPartitions uint32              `json:"maxNumPartitions"`
	PartitionInfo    []fmpmPartitionInfo `json:"partitionInfo"`
}

type fmpmPartitionInfo struct {
	PartitionID uint32        `json:"partitionId"`
	IsActive    uint32        `json:"isActive"`
	NumGpus     uint32        `json:"numGpus"`
	GpuInfo     []fmpmGpuInfo `json:"gpuInfo"`
}

type fmpmGpuInfo struct {
	PhysicalID          uint32 `json:"physicalId"`
	UUID                string `json:"uuid"`
	PCIBusID            string `json:"pciBusId"`
	NumNvLinksAvailable uint32 `json:"numNvLinksAvailable"`
	MaxNumNvLinks       uint32 `json:"maxNumNvLinks"`
	NvLinkLineRateMBps  uint32 `json:"nvlinkLineRateMBps"`
}

func (l fmpmPartitionList) toPartitions() []Partition {
	out := make([]Partition, 0, len(l.PartitionInfo))
	for _, p := range l.PartitionInfo {
		gpus := make([]PartitionGPU, 0, len(p.GpuInfo))
		for _, g := range p.GpuInfo {
			gpus = append(gpus, PartitionGPU{
				PhysicalID:          int(g.PhysicalID),
				UUID:                g.UUID,
				PCIBusID:            g.PCIBusID,
				NumNvLinksAvailable: g.NumNvLinksAvailable,
				MaxNumNvLinks:       g.MaxNumNvLinks,
				NvLinkLineRateMBps:  g.NvLinkLineRateMBps,
			})
		}
		out = append(out, Partition{
			ID:       int(p.PartitionID),
			IsActive: p.IsActive != 0,
			GPUs:     gpus,
		})
	}
	return out
}

type fmpmUnsupportedPartitionList struct {
	NumPartitions uint32                         `json:"numPartitions"`
	PartitionInfo []fmpmUnsupportedPartitionInfo `json:"partitionInfo"`
}

type fmpmUnsupportedPartitionInfo struct {
	PartitionID    uint32   `json:"partitionId"`
	NumGpus        uint32   `json:"numGpus"`
	GpuPhysicalIDs []uint32 `json:"gpuPhysicalIds"`
}

func (l fmpmUnsupportedPartitionList) toUnsupportedPartitions() []UnsupportedPartition {
	out := make([]UnsupportedPartition, 0, len(l.PartitionInfo))
	for _, p := range l.PartitionInfo {
		ids := make([]int, 0, len(p.GpuPhysicalIDs))
		for _, id := range p.GpuPhysicalIDs {
			ids = append(ids, int(id))
		}
		out = append(out, UnsupportedPartition{
			ID:             int(p.PartitionID),
			GPUPhysicalIDs: ids,
		})
	}
	return out
}

type fmpmNvlinkFailedDevices struct {
	NumGpus     uint32                       `json:"numGpus"`
	NumSwitches uint32                       `json:"numSwitches"`
	GpuInfo     []fmpmNvlinkFailedDeviceInfo `json:"gpuInfo"`
	SwitchInfo  []fmpmNvlinkFailedDeviceInfo `json:"switchInfo"`
}

type fmpmNvlinkFailedDeviceInfo struct {
	UUID     string   `json:"uuid"`
	PCIBusID string   `json:"pciBusId"`
	NumPorts uint32   `json:"numPorts"`
	PortNum  []uint32 `json:"portNum"`
}

func (d fmpmNvlinkFailedDevices) toNvlinkFailedDevices() NvlinkFailedDevices {
	conv := func(infos []fmpmNvlinkFailedDeviceInfo) []NvlinkFailedDevice {
		out := make([]NvlinkFailedDevice, 0, len(infos))
		for _, info := range infos {
			ports := make([]uint32, len(info.PortNum))
			copy(ports, info.PortNum)
			out = append(out, NvlinkFailedDevice{
				UUID:     info.UUID,
				PCIBusID: info.PCIBusID,
				Ports:    ports,
			})
		}
		return out
	}
	return NvlinkFailedDevices{
		GPUs:     conv(d.GpuInfo),
		Switches: conv(d.SwitchInfo),
	}
}
