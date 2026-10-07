// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

package pithjpeg

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// repoRoot resolves the repository root relative to this package
// (sdk/go -> sdk -> repo root), the anchor for reference.json and the
// committed fixtures.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(filepath.Join(root, "reference.json")); err != nil || st.IsDir() {
		t.Fatalf("reference.json not found at %s", root)
	}
	return root
}

// jpegReference mirrors the committed reference.json: the digests
// section plus the four numeric vectors.
type jpegReference struct {
	Digests map[string]string `json:"digests"`
	Vectors map[string]struct {
		Input  []string `json:"input"`
		Output []string `json:"output"`
		Exact  bool     `json:"exact"`
		TolAbs string   `json:"tol_abs"`
		TolRel string   `json:"tol_rel"`
	} `json:"vectors"`
}

// reference parses the committed reference.json.
func reference(t *testing.T) jpegReference {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref jpegReference
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	return ref
}

// fixture reads one committed fixture file.
func fixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// lanesFromHex decodes the reference file's big-endian hex lanes.
func lanesFromHex(hexes []string) []float64 {
	out := make([]float64, len(hexes))
	for i, h := range hexes {
		out[i] = laneValue(h)
	}
	return out
}

func TestFindCdylib(t *testing.T) {
	p, err := FindCdylib()
	if err != nil {
		t.Skipf("no cdylib built: %v", err)
	}
	if filepath.Ext(p) != ".dll" && filepath.Ext(p) != ".so" && filepath.Ext(p) != ".dylib" {
		t.Fatalf("unexpected cdylib path %s", p)
	}
}

func TestEveryCommittedDigestIsReproduced(t *testing.T) {
	ref := reference(t)
	for _, name := range sortedKeys(ref.Digests) {
		t.Run(name, func(t *testing.T) {
			data := fixture(t, name+".jpg")
			channels, err := Channels(data)
			if err != nil {
				t.Fatal(err)
			}
			layout := LayoutRgb8
			if channels == 1 {
				layout = LayoutGray8
			}
			got, err := DecodeDigest(data, layout)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprintf("%016x", got) != ref.Digests[name] {
				t.Fatalf("%s: digest %016x != %s", name, got, ref.Digests[name])
			}
		})
	}
}

func TestPinnedLiteralBaseGray(t *testing.T) {
	got, err := DecodeDigest(fixture(t, "base_gray.jpg"), LayoutGray8)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0xe0ce77f00e2cb066 {
		t.Fatalf("pinned base_gray digest drifted: %016x", got)
	}
}

func TestNativeChannelsReport(t *testing.T) {
	got, err := Channels(fixture(t, "base_gray.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("gray channels = %d", got)
	}
	got, err = Channels(fixture(t, "base_444.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if got != 3 {
		t.Fatalf("444 channels = %d", got)
	}
}

func TestLayoutConversionsWiden(t *testing.T) {
	data := fixture(t, "base_gray.jpg")
	seen := map[uint64]bool{}
	for _, layout := range []uint32{LayoutGray8, LayoutRgb8, LayoutRgba8} {
		got, err := DecodeDigest(data, layout)
		if err != nil {
			t.Fatal(err)
		}
		if seen[got] {
			t.Fatalf("layout %d digest collides with an earlier layout", layout)
		}
		seen[got] = true
	}
}

func TestRefusals(t *testing.T) {
	if _, err := DecodeDigest(fixture(t, "base_444.jpg"), LayoutGray8); !isRejected(err) {
		t.Fatalf("rgb to gray = %v, want rejected", err)
	}
	if _, err := DecodeDigest([]byte("not a jpeg at all"), LayoutGray8); !isRejected(err) {
		t.Fatalf("garbage = %v, want rejected", err)
	}
	if _, err := DecodeDigest(nil, LayoutGray8); !isRejected(err) {
		t.Fatalf("empty stream = %v, want rejected", err)
	}
	if _, err := DecodeDigest(fixture(t, "base_gray.jpg"), 9); !isInvalid(err) {
		t.Fatalf("unknown layout = %v, want invalid", err)
	}
	if _, err := Channels([]byte("x")); !isRejected(err) {
		t.Fatalf("garbage channels = %v, want rejected", err)
	}
}

func isRejected(err error) bool {
	e, ok := err.(*FfiError)
	return ok && e.Status == StatusRejected
}

func isInvalid(err error) bool {
	e, ok := err.(*FfiError)
	return ok && e.Status == StatusInvalid
}

func TestEveryKernelVectorIsReproduced(t *testing.T) {
	ref := reference(t)
	for _, name := range sortedKeys(ref.Vectors) {
		t.Run(name, func(t *testing.T) {
			vector := ref.Vectors[name]
			input := lanesFromHex(vector.Input)
			var (
				got []float64
				err error
			)
			switch name {
			case "dequant.zigzag.8x8":
				got, err = DequantZigzag(input)
			case "dequant.8x8":
				got, err = Dequant(input[:Block], input[Block:])
			case "idct.islow.8x8":
				got, err = IdctIslow(input[:Block], input[Block:])
			case "idct.oracle.8x8":
				got, err = IdctOracle(input)
			default:
				t.Fatalf("unmapped vector %s", name)
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(vector.Output) {
				t.Fatalf("output lane count %d != %d", len(got), len(vector.Output))
			}
			for i := range got {
				want := laneValue(vector.Output[i])
				if vector.Exact {
					if math.Float64bits(got[i]) != math.Float64bits(want) {
						t.Fatalf("%s lane %d: %v != %v", name, i, got[i], want)
					}
				} else {
					tolAbs := laneValue(vector.TolAbs)
					tolRel := laneValue(vector.TolRel)
					budget := math.Max(tolAbs, tolRel*math.Abs(want))
					if math.Abs(got[i]-want) > budget {
						t.Fatalf("%s lane %d: %v vs %v (budget %v)", name, i, got[i], want, budget)
					}
				}
			}
		})
	}
}

func TestKernelLengthValidation(t *testing.T) {
	if _, err := DequantZigzag(make([]float64, Block-1)); err == nil {
		t.Fatal("short zigzag accepted")
	}
	if _, err := Dequant(make([]float64, Block), make([]float64, Block-1)); err == nil {
		t.Fatal("short dequant accepted")
	}
	if _, err := IdctOracle(make([]float64, Block+1)); err == nil {
		t.Fatal("long oracle accepted")
	}
}

// sortedKeys returns the map keys in sorted order for deterministic
// subtests.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := range len(keys) - 1 {
		for j := range len(keys) - 1 - i {
			if keys[j] > keys[j+1] {
				keys[j], keys[j+1] = keys[j+1], keys[j]
			}
		}
	}
	return keys
}
