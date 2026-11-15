package rbtree

import (
	"math/rand"
	"sort"
	"testing"
)

func TestRandomInsertDelete(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tr := New(func(a, b int) bool { return a < b })
	var nodes []*Node[int]
	var want []int
	for i := 0; i < 5000; i++ {
		if len(nodes) > 0 && rng.Intn(3) == 0 {
			j := rng.Intn(len(nodes))
			v := nodes[j].Value
			tr.Delete(nodes[j])
			nodes = append(nodes[:j], nodes[j+1:]...)
			k := sort.SearchInts(want, v)
			want = append(want[:k], want[k+1:]...)
		} else {
			v := rng.Intn(1000)
			nodes = append(nodes, tr.Insert(v))
			k := sort.SearchInts(want, v)
			want = append(want, 0)
			copy(want[k+1:], want[k:])
			want[k] = v
		}
		tr.check()
		if tr.Len() != len(want) {
			t.Fatalf("len %d, want %d", tr.Len(), len(want))
		}
	}
	var got []int
	for n := tr.Min(); n != nil; n = tr.Next(n) {
		got = append(got, n.Value)
	}
	if len(got) != len(want) {
		t.Fatalf("walk len %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("at %d got %d want %d", i, got[i], want[i])
		}
	}
}

func TestDoubleDeleteIsNoop(t *testing.T) {
	tr := New(func(a, b int) bool { return a < b })
	n := tr.Insert(1)
	tr.Insert(2)
	tr.Delete(n)
	tr.Delete(n)
	if tr.Len() != 1 || tr.Min().Value != 2 {
		t.Fatalf("unexpected tree state")
	}
}
