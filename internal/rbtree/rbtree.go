// Package rbtree is a generic red-black tree (CLRS style, with a sentinel),
// used by the EEVDF scheduler the same way Linux keeps its runqueue.
package rbtree

type color bool

const (
	red   color = true
	black color = false
)

// Node is an element of the tree. Keep the pointer returned by Insert to
// delete the element in O(log n).
type Node[T any] struct {
	Value               T
	left, right, parent *Node[T]
	color               color
	tree                *Tree[T]
}

// Tree is an ordered multiset. Equal elements are kept in insertion order.
type Tree[T any] struct {
	root *Node[T]
	nil_ *Node[T]
	less func(a, b T) bool
	size int
}

// New returns an empty tree ordered by less.
func New[T any](less func(a, b T) bool) *Tree[T] {
	s := &Node[T]{color: black}
	return &Tree[T]{root: s, nil_: s, less: less}
}

// Len returns the number of elements.
func (t *Tree[T]) Len() int { return t.size }

// Insert adds v and returns its node.
func (t *Tree[T]) Insert(v T) *Node[T] {
	z := &Node[T]{Value: v, left: t.nil_, right: t.nil_, color: red, tree: t}
	y := t.nil_
	x := t.root
	for x != t.nil_ {
		y = x
		if t.less(z.Value, x.Value) {
			x = x.left
		} else {
			x = x.right
		}
	}
	z.parent = y
	switch {
	case y == t.nil_:
		t.root = z
	case t.less(z.Value, y.Value):
		y.left = z
	default:
		y.right = z
	}
	t.insertFixup(z)
	t.size++
	return z
}

// Min returns the smallest node, or nil if the tree is empty.
func (t *Tree[T]) Min() *Node[T] {
	if t.root == t.nil_ {
		return nil
	}
	return t.minimum(t.root)
}

// Next returns the in-order successor of n, or nil.
func (t *Tree[T]) Next(n *Node[T]) *Node[T] {
	if n.right != t.nil_ {
		return t.minimum(n.right)
	}
	y := n.parent
	for y != t.nil_ && n == y.right {
		n = y
		y = y.parent
	}
	if y == t.nil_ {
		return nil
	}
	return y
}

// Delete removes n from the tree. Deleting a node that is not in this tree
// (or was already deleted) is a no-op.
func (t *Tree[T]) Delete(z *Node[T]) {
	if z == nil || z.tree != t {
		return
	}
	y := z
	yOrig := y.color
	var x *Node[T]
	switch {
	case z.left == t.nil_:
		x = z.right
		t.transplant(z, z.right)
	case z.right == t.nil_:
		x = z.left
		t.transplant(z, z.left)
	default:
		y = t.minimum(z.right)
		yOrig = y.color
		x = y.right
		if y.parent == z {
			x.parent = y
		} else {
			t.transplant(y, y.right)
			y.right = z.right
			y.right.parent = y
		}
		t.transplant(z, y)
		y.left = z.left
		y.left.parent = y
		y.color = z.color
	}
	if yOrig == black {
		t.deleteFixup(x)
	}
	t.size--
	z.left, z.right, z.parent, z.tree = nil, nil, nil, nil
}

func (t *Tree[T]) minimum(x *Node[T]) *Node[T] {
	for x.left != t.nil_ {
		x = x.left
	}
	return x
}

func (t *Tree[T]) leftRotate(x *Node[T]) {
	y := x.right
	x.right = y.left
	if y.left != t.nil_ {
		y.left.parent = x
	}
	y.parent = x.parent
	switch {
	case x.parent == t.nil_:
		t.root = y
	case x == x.parent.left:
		x.parent.left = y
	default:
		x.parent.right = y
	}
	y.left = x
	x.parent = y
}

func (t *Tree[T]) rightRotate(x *Node[T]) {
	y := x.left
	x.left = y.right
	if y.right != t.nil_ {
		y.right.parent = x
	}
	y.parent = x.parent
	switch {
	case x.parent == t.nil_:
		t.root = y
	case x == x.parent.right:
		x.parent.right = y
	default:
		x.parent.left = y
	}
	y.right = x
	x.parent = y
}

func (t *Tree[T]) insertFixup(z *Node[T]) {
	for z.parent.color == red {
		if z.parent == z.parent.parent.left {
			y := z.parent.parent.right
			if y.color == red {
				z.parent.color = black
				y.color = black
				z.parent.parent.color = red
				z = z.parent.parent
			} else {
				if z == z.parent.right {
					z = z.parent
					t.leftRotate(z)
				}
				z.parent.color = black
				z.parent.parent.color = red
				t.rightRotate(z.parent.parent)
			}
		} else {
			y := z.parent.parent.left
			if y.color == red {
				z.parent.color = black
				y.color = black
				z.parent.parent.color = red
				z = z.parent.parent
			} else {
				if z == z.parent.left {
					z = z.parent
					t.rightRotate(z)
				}
				z.parent.color = black
				z.parent.parent.color = red
				t.leftRotate(z.parent.parent)
			}
		}
	}
	t.root.color = black
}

func (t *Tree[T]) transplant(u, v *Node[T]) {
	switch {
	case u.parent == t.nil_:
		t.root = v
	case u == u.parent.left:
		u.parent.left = v
	default:
		u.parent.right = v
	}
	v.parent = u.parent
}

func (t *Tree[T]) deleteFixup(x *Node[T]) {
	for x != t.root && x.color == black {
		if x == x.parent.left {
			w := x.parent.right
			if w.color == red {
				w.color = black
				x.parent.color = red
				t.leftRotate(x.parent)
				w = x.parent.right
			}
			if w.left.color == black && w.right.color == black {
				w.color = red
				x = x.parent
			} else {
				if w.right.color == black {
					w.left.color = black
					w.color = red
					t.rightRotate(w)
					w = x.parent.right
				}
				w.color = x.parent.color
				x.parent.color = black
				w.right.color = black
				t.leftRotate(x.parent)
				x = t.root
			}
		} else {
			w := x.parent.left
			if w.color == red {
				w.color = black
				x.parent.color = red
				t.rightRotate(x.parent)
				w = x.parent.left
			}
			if w.right.color == black && w.left.color == black {
				w.color = red
				x = x.parent
			} else {
				if w.left.color == black {
					w.right.color = black
					w.color = red
					t.leftRotate(w)
					w = x.parent.left
				}
				w.color = x.parent.color
				x.parent.color = black
				w.left.color = black
				t.rightRotate(x.parent)
				x = t.root
			}
		}
	}
	x.color = black
}

// check verifies the red-black invariants and returns the black height. It
// panics on violation. Used by tests.
func (t *Tree[T]) check() int {
	if t.root.color != black {
		panic("root is red")
	}
	var walk func(n *Node[T]) int
	walk = func(n *Node[T]) int {
		if n == t.nil_ {
			return 1
		}
		if n.color == red && (n.left.color == red || n.right.color == red) {
			panic("red node with red child")
		}
		if n.left != t.nil_ && (n.left.parent != n || t.less(n.Value, n.left.Value)) {
			panic("bad left child")
		}
		if n.right != t.nil_ && (n.right.parent != n || t.less(n.right.Value, n.Value)) {
			panic("bad right child")
		}
		l, r := walk(n.left), walk(n.right)
		if l != r {
			panic("unequal black height")
		}
		if n.color == black {
			l++
		}
		return l
	}
	return walk(t.root)
}
