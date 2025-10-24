// Copyright 2025 The Auto Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

//go:embed books/*
var Text embed.FS

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

const (
	order = 4
)

type Markov [order]byte
type Model [order]map[Markov][]uint32

// Lookup looks a vector up
func Lookup(markov *[order]Markov, model *Model) []float32 {
	for i := range markov {
		i = order - 1 - i
		vector := model[i][markov[i]]
		if vector != nil {
			sum := float32(0.0)
			for _, value := range vector {
				sum += float32(value)
			}
			result := make([]float32, len(vector))
			for ii, value := range vector {
				result[ii] = float32(value) / sum
			}
			return result
		}
	}
	return nil
}

// Iterate iterates a markov model
func Iterate(markov *[order]Markov, state byte) {
	for i := range markov {
		state := state
		for ii, value := range markov[i][:i+1] {
			markov[i][ii], state = state, value
		}
	}
}

func main() {
	type File struct {
		Name  string
		Data  []byte
		Model Model
	}

	files := []File{
		{Name: "10.txt.utf-8.bz2"},
		{Name: "pg74.txt.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}

	load := func(book *File) {
		path := fmt.Sprintf("books/%s", book.Name)
		file, err := Text.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}

		markov := [order]Markov{}
		for i := range book.Model {
			book.Model[i] = make(map[Markov][]uint32)
		}
		for _, value := range data {
			for ii := range markov {
				vector := book.Model[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, 256)
				}
				vector[value]++
				book.Model[ii][markov[ii]] = vector

				state := value
				for iii, value := range markov[ii][:ii+1] {
					markov[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}

	for i := range files {
		load(&files[i])
		fmt.Println(files[i].Name)
	}

	type Auto struct {
		Set       tf64.Set
		Iteration int
	}

	rng := rand.New(rand.NewSource(1))

	autos := make([]Auto, 256)
	for i := range autos {
		autos[i].Set = tf64.NewSet()
		autos[i].Set.Add("l1", 256, 256)
		autos[i].Set.Add("b1", 256, 1)
		autos[i].Set.Add("l2", 512, 256)
		autos[i].Set.Add("b2", 256, 1)

		for ii := range autos[i].Set.Weights {
			w := autos[i].Set.Weights[ii]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
		}
	}

	//histogram := NewHistogram(33)
	markov := [order]Markov{}
	iteration := 0

	//histogram.Add(0)
	Iterate(&markov, 0)
	for _, value := range files[0].Data[:256*1024] {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(autos[value].Iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		others := tf64.NewSet()
		others.Add("input", 256, 1)
		others.Add("output", 256, 1)
		in := others.ByName["input"]
		out := others.ByName["output"]
		/*sum := 0
		for _, v := range histogram.Vector {
			sum += int(v)
		}
		for _, v := range histogram.Vector {
			vv := float64(v) / float64(sum)
			in.X = append(in.X, vv)
			out.X = append(out.X, vv)
		}*/
		vector := Lookup(&markov, &files[0].Model)
		for _, v := range vector {
			in.X = append(in.X, float64(v))
			out.X = append(out.X, float64(v))
		}

		l1 := tf64.Everett(tf64.Add(tf64.Mul(autos[value].Set.Get("l1"), others.Get("input")), autos[value].Set.Get("b1")))
		l2 := tf64.Add(tf64.Mul(autos[value].Set.Get("l2"), l1), autos[value].Set.Get("b2"))
		loss := tf64.Sum(tf64.Quadratic(l2, others.Get("output")))

		l := 0.0
		autos[value].Set.Zero()
		others.Zero()
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range autos[value].Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range autos[value].Set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		iteration++
		autos[value].Iteration++
		if iteration%1024 == 0 || iteration < 1024 {
			fmt.Println(iteration, l)
		}

		//histogram.Add(value)
		Iterate(&markov, value)
	}

	prompt := "What is the meaning of life?"
	str := []byte(prompt)
	//histogram = NewHistogram(33)
	markov = [order]Markov{}
	for _, value := range str {
		//histogram.Add(value)
		Iterate(&markov, value)
	}
	for range 33 {
		distribution := make([]float64, len(autos))
		for i := range autos {
			others := tf64.NewSet()
			others.Add("input", 256, 1)
			others.Add("output", 256, 1)
			in := others.ByName["input"]
			out := others.ByName["output"]
			/*sum := 0
			for _, v := range histogram.Vector {
				sum += int(v)
			}
			for _, v := range histogram.Vector {
				vv := float64(v) / float64(sum)
				in.X = append(in.X, vv)
				out.X = append(out.X, vv)
			}*/
			vector := Lookup(&markov, &files[0].Model)
			for _, v := range vector {
				in.X = append(in.X, float64(v))
				out.X = append(out.X, float64(v))
			}
			l1 := tf64.Everett(tf64.Add(tf64.Mul(autos[i].Set.Get("l1"), others.Get("input")), autos[i].Set.Get("b1")))
			l2 := tf64.Add(tf64.Mul(autos[i].Set.Get("l2"), l1), autos[i].Set.Get("b2"))
			loss := tf64.Sum(tf64.Quadratic(l2, others.Get("output")))

			autos[i].Set.Zero()
			others.Zero()
			loss(func(a *tf64.V) bool {
				distribution[i] = a.X[0]
				return true
			})
		}
		max := 0.0
		for _, value := range distribution {
			if value > max {
				max = value
			}
		}
		sum := 0.0
		for i, value := range distribution {
			value = max - value
			sum += value
			distribution[i] = value
		}
		total, selected := 0.0, rng.Float64()
		for i, value := range distribution {
			total += value / sum
			if selected < total {
				str = append(str, byte(i))
				//histogram.Add(byte(i))
				Iterate(&markov, byte(i))
				break
			}
		}
	}
	fmt.Println(string(str))
}
