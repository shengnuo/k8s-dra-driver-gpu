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
	"errors"
	"reflect"
	"testing"
)

// fakeRunner records the argument vectors it is invoked with and returns
// canned output/error, standing in for the fmpm binary.
type fakeRunner struct {
	calls  [][]string
	out    []byte
	err    error
	// outByFirstArg, when non-nil, selects the response by the first
	// non-flag argument (e.g. "-l"), allowing multiple ops in one test.
	outByFirstArg map[string][]byte
}

func (r *fakeRunner) run(args []string) ([]byte, error) {
	r.calls = append(r.calls, args)
	if r.outByFirstArg != nil {
		for _, a := range args {
			if out, ok := r.outByFirstArg[a]; ok {
				return out, r.err
			}
		}
	}
	return r.out, r.err
}

// newTestClient builds an fmpmClient wired to the given fake runner and marks
// it connected so query/activation methods proceed past the guard.
func newTestClient(r *fakeRunner) *fmpmClient {
	return &fmpmClient{
		binaryPath: "fmpm",
		connected:  true,
		runner:     r.run,
	}
}

const samplePartitionsJSON = `{
   "version" : 1,
   "numPartitions" : 2,
   "maxNumPartitions" : 8,
   "partitionInfo" : [
      {
         "partitionId" : 0,
         "isActive" : 1,
         "numGpus" : 1,
         "gpuInfo" : [
            {
               "physicalId" : 3,
               "uuid" : "GPU-aaaa",
               "pciBusId" : "00000000:3B:00.0",
               "numNvLinksAvailable" : 18,
               "maxNumNvLinks" : 18,
               "nvlinkLineRateMBps" : 25000
            }
         ]
      },
      {
         "partitionId" : 7,
         "isActive" : 0,
         "numGpus" : 2,
         "gpuInfo" : [
            {
               "physicalId" : 0,
               "uuid" : "GPU-bbbb",
               "pciBusId" : "00000000:1B:00.0",
               "numNvLinksAvailable" : 18,
               "maxNumNvLinks" : 18,
               "nvlinkLineRateMBps" : 25000
            },
            {
               "physicalId" : 1,
               "uuid" : "GPU-cccc",
               "pciBusId" : "00000000:2B:00.0",
               "numNvLinksAvailable" : 18,
               "maxNumNvLinks" : 18,
               "nvlinkLineRateMBps" : 25000
            }
         ]
      }
   ]
}`

func TestFMPMGetSupportedFabricPartitions(t *testing.T) {
	r := &fakeRunner{out: []byte(samplePartitionsJSON)}
	c := newTestClient(r)

	parts, err := c.GetSupportedFabricPartitions()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []Partition{
		{
			ID:       0,
			IsActive: true,
			GPUs: []PartitionGPU{
				{PhysicalID: 3, UUID: "GPU-aaaa", PCIBusID: "00000000:3B:00.0", NumNvLinksAvailable: 18, MaxNumNvLinks: 18, NvLinkLineRateMBps: 25000},
			},
		},
		{
			ID:       7,
			IsActive: false,
			GPUs: []PartitionGPU{
				{PhysicalID: 0, UUID: "GPU-bbbb", PCIBusID: "00000000:1B:00.0", NumNvLinksAvailable: 18, MaxNumNvLinks: 18, NvLinkLineRateMBps: 25000},
				{PhysicalID: 1, UUID: "GPU-cccc", PCIBusID: "00000000:2B:00.0", NumNvLinksAvailable: 18, MaxNumNvLinks: 18, NvLinkLineRateMBps: 25000},
			},
		},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("partitions mismatch:\n got: %+v\nwant: %+v", parts, want)
	}

	if len(r.calls) != 1 || !reflect.DeepEqual(r.calls[0], []string{"-l"}) {
		t.Fatalf("unexpected fmpm calls: %v", r.calls)
	}
}

func TestFMPMConnectArgs(t *testing.T) {
	tests := []struct {
		name   string
		params ConnectParams
		op     string
		want   []string
	}{
		{
			name:   "default address",
			params: ConnectParams{},
			op:     "-l",
			want:   []string{"-l"},
		},
		{
			name:   "tcp hostname",
			params: ConnectParams{AddressInfo: "192.168.0.1:6666"},
			op:     "-l",
			want:   []string{"--hostname", "192.168.0.1:6666", "-l"},
		},
		{
			name:   "unix socket",
			params: ConnectParams{AddressInfo: "/run/fm.sock", AddressIsUnixSocket: true},
			op:     "-l",
			want:   []string{"--unix-domain-socket", "/run/fm.sock", "-l"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &fakeRunner{out: []byte(`{"numPartitions":0,"partitionInfo":[]}`)}
			c := newTestClient(r)
			if err := c.Connect(tt.params); err != nil {
				t.Fatalf("connect: %v", err)
			}
			if _, err := c.GetSupportedFabricPartitions(); err != nil {
				t.Fatalf("list: %v", err)
			}
			if len(r.calls) != 1 || !reflect.DeepEqual(r.calls[0], tt.want) {
				t.Fatalf("args mismatch:\n got: %v\nwant: %v", r.calls, tt.want)
			}
		})
	}
}

func TestFMPMActivateDeactivate(t *testing.T) {
	r := &fakeRunner{}
	c := newTestClient(r)

	if err := c.ActivateFabricPartition(7); err != nil {
		t.Fatalf("activate: %v", err)
	}
	if err := c.DeactivateFabricPartition(7); err != nil {
		t.Fatalf("deactivate: %v", err)
	}

	want := [][]string{
		{"-a", "7"},
		{"-d", "7"},
	}
	if !reflect.DeepEqual(r.calls, want) {
		t.Fatalf("calls mismatch:\n got: %v\nwant: %v", r.calls, want)
	}
}

func TestFMPMSetActivatedFabricPartitions(t *testing.T) {
	r := &fakeRunner{}
	c := newTestClient(r)

	if err := c.SetActivatedFabricPartitions([]int{2, 0, 5}); err != nil {
		t.Fatalf("set activated: %v", err)
	}
	want := []string{"--set-activated-list", "2,0,5"}
	if len(r.calls) != 1 || !reflect.DeepEqual(r.calls[0], want) {
		t.Fatalf("args mismatch:\n got: %v\nwant: %v", r.calls, want)
	}
}

func TestFMPMGetUnsupportedFabricPartitions(t *testing.T) {
	const js = `{
       "version" : 1,
       "numPartitions" : 1,
       "partitionInfo" : [
          {
             "partitionId" : 3,
             "numGpus" : 2,
             "gpuPhysicalIds" : [ 4, 6 ]
          }
       ]
    }`
	r := &fakeRunner{out: []byte(js)}
	c := newTestClient(r)

	got, err := c.GetUnsupportedFabricPartitions()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []UnsupportedPartition{{ID: 3, GPUPhysicalIDs: []int{4, 6}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mismatch:\n got: %+v\nwant: %+v", got, want)
	}
	if len(r.calls) != 1 || r.calls[0][0] != "--list-unsupported-partitions" {
		t.Fatalf("unexpected call: %v", r.calls)
	}
}

func TestFMPMGetNvlinkFailedDevices(t *testing.T) {
	const js = `{
       "version" : 1,
       "numGpus" : 1,
       "numSwitches" : 1,
       "gpuInfo" : [
          { "uuid" : "GPU-x", "pciBusId" : "00000000:3B:00.0", "numPorts" : 2, "portNum" : [ 0, 5 ] }
       ],
       "switchInfo" : [
          { "uuid" : "SW-y", "pciBusId" : "00000000:4B:00.0", "numPorts" : 1, "portNum" : [ 12 ] }
       ]
    }`
	r := &fakeRunner{out: []byte(js)}
	c := newTestClient(r)

	got, err := c.GetNvlinkFailedDevices()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := NvlinkFailedDevices{
		GPUs:     []NvlinkFailedDevice{{UUID: "GPU-x", PCIBusID: "00000000:3B:00.0", Ports: []uint32{0, 5}}},
		Switches: []NvlinkFailedDevice{{UUID: "SW-y", PCIBusID: "00000000:4B:00.0", Ports: []uint32{12}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mismatch:\n got: %+v\nwant: %+v", got, want)
	}
}

func TestFMPMActivateWithVFsUnsupported(t *testing.T) {
	r := &fakeRunner{}
	c := newTestClient(r)
	err := c.ActivateFabricPartitionWithVFs(1, []PCIDevice{{}})
	if err == nil {
		t.Fatal("expected error for unsupported VF activation")
	}
	if len(r.calls) != 0 {
		t.Fatalf("expected no fmpm invocation, got: %v", r.calls)
	}
}

func TestFMPMNotConnected(t *testing.T) {
	r := &fakeRunner{out: []byte(samplePartitionsJSON)}
	c := &fmpmClient{binaryPath: "fmpm", runner: r.run} // connected == false

	if _, err := c.GetSupportedFabricPartitions(); !errors.Is(err, errNotConnected) {
		t.Fatalf("expected errNotConnected, got %v", err)
	}
	if err := c.ActivateFabricPartition(1); !errors.Is(err, errNotConnected) {
		t.Fatalf("expected errNotConnected, got %v", err)
	}
	if len(r.calls) != 0 {
		t.Fatalf("expected no fmpm invocation before connect, got: %v", r.calls)
	}
}

func TestFMPMExitErrorPropagated(t *testing.T) {
	// fmpm returns the fmReturn_t code as its exit status; the wrapper
	// surfaces it via *fmpmExitError.
	r := &fakeRunner{err: &fmpmExitError{Args: []string{"-a", "7"}, Code: 13, Stdout: "Failed to activate partition. fmReturn: 13"}}
	c := newTestClient(r)

	err := c.ActivateFabricPartition(7)
	if err == nil {
		t.Fatal("expected error")
	}
	var exitErr *fmpmExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected *fmpmExitError, got %T: %v", err, err)
	}
	if exitErr.Code != 13 {
		t.Fatalf("expected exit code 13, got %d", exitErr.Code)
	}
}

func TestFMPMInvalidJSON(t *testing.T) {
	r := &fakeRunner{out: []byte("not json")}
	c := newTestClient(r)
	if _, err := c.GetSupportedFabricPartitions(); err == nil {
		t.Fatal("expected JSON parse error")
	}
}

// Ensure the fmpm client satisfies the Client interface.
var _ Client = (*fmpmClient)(nil)
