// Copyright 2023 FortAI-Hub contributors.
// Released under the MIT License. See LICENSE file for full license information.

package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

func main() {
	if len(os.Args) < 2 {
		panic("missing model filename")
	}
	if err := convert(os.Args[1]); err != nil {
		panic(err)
	}
}

type array1D struct {
	data []float32
}

func (a *array1D) write(w io.Writer) error {
	if err := binary.Write(w, binary.LittleEndian, a.data); err != nil {
		return err
	}
	return nil
}

type array2D struct {
	transposed bool
	rows       int
	cols       int
	data       []float32
}

func (a *array2D) write(w io.Writer) error {
	if err := binary.Write(w, binary.LittleEndian, t(a.data, a.rows)); err != nil {
		return err
	}
	return nil
}

// t returns the transpose of the matrix.
func t(data []float32, rows int) []float32 {
	out := make([]float32, len(data))
	size := len(data)
	index := 0
	for _, value := range data {
		out[index] = value
		index += rows
		if index >= size {
			index -= size - 1
		}
	}
	return out
}

type layerNormType struct {
	g array1D
	b array1D
}

func (l *layerNormType) write(w io.Writer) error {
	if err := l.g.write(w); err != nil {
		return err
	}
	if err := l.b.write(w); err != nil {
		return err
	}
	return nil
}

type channelMixType struct {
	wk   array2D
	wv   array2D
	wr   array2D
	mixK array1D
	mixR array1D
}

func (c *channelMixType) write(w io.Writer) error {
	if err := c.wk.write(w); err != nil {
		return err
	}
	if err := c.wv.write(w); err != nil {
		return err
	}
	if err := c.wr.write(w); err != nil {
		return err
	}
	if err := c.mixK.write(w); err != nil {
		return err
	}
	if err := c.mixR.write(w); err != nil {
		return err
	}
	return nil
}

type timeMixType struct {
	wk        array2D
	wv        array2D
	wr        array2D
	wo        array2D
	mixK      array1D
	mixV      array1D
	mixR      array1D
	timeDecay array1D
	timeFirst array1D
}

func (t *timeMixType) write(w io.Writer) error {
	if err := t.wk.write(w); err != nil {
		return err
	}
	if err := t.wv.write(w); err != nil {
		return err
	}
	if err := t.wr.write(w); err != nil {
		return err
	}
	if err := t.wo.write(w); err != nil {
		return err
	}
	if err := t.mixK.write(w); err != nil {
		return err
	}
	if err := t.mixV.write(w); err != nil {
		return err
	}
	if err := t.mixR.write(w); err != nil {
		return err
	}
	if err := t.timeDecay.write(w); err != nil {
		return err
	}
	if err := t.timeFirst.write(w); err != nil {
		return err
	}
	return nil
}

type rwkvLayerType struct {
	ln1        layerNormType
	ln2        layerNormType
	channelMix channelMixType
	timeMix    timeMixType
}

func (r *rwkvLayerType) write(w io.Writer) error {
	if err := r.ln1.write(w); err != nil {
		return err
	}
	if err := r.ln2.write(w); err != nil {
		return err
	}
	if err := r.channelMix.write(w); err != nil {
		return err
	}
	if err := r.timeMix.write(w); err != nil {
		return err
	}
	return nil
}

type rwkvType struct {
	ln     layerNormType
	layers []rwkvLayerType
}

func (r *rwkvType) write(w io.Writer) error {
	if err := r.ln.write(w); err != nil {
		return err
	}
	for _, layer := range r.layers {
		if err := layer.write(w); err != nil {
			return err
		}
	}
	return nil
}

type rwkvLmType struct {
	rwkv rwkvType
	ln   layerNormType
	emb  array2D
	proj array2D
}

func (r *rwkvLmType) write(w io.Writer) error {
	if err := r.emb.write(w); err != nil {
		return err
	}
	if err := r.rwkv.write(w); err != nil {
		return err
	}
	if err := r.ln.write(w); err != nil {
		return err
	}
	if err := r.proj.write(w); err != nil {
		return err
	}
	return nil
}

func convert(filename string) error {
	c := converter{
		inFilename: filename,
	}
	return c.run()
}

type converter struct {
	inFilename string
	nLayers    int
	dm         int
	vocabSize  int
	model      rwkvLmType
	params     paramsMap
}

func (c *converter) run() error {
	funcs := []func() error{
		c.loadTorchModelParams,
		c.convEmbeddings,
		c.convLinear,
		c.convOutLayerNorm,
		c.convRWKVLayerNorm,
		c.convBlocks,
		c.writeModel,
	}
	for _, fn := range funcs {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}

func (c *converter) writeModel() (err error) {
	outFilename := c.inFilename + ".bin"
	f, err := os.Create(outFilename)
	if err != nil {
		return
	}
	defer func() {
		if e := f.Close(); e != nil {
			err = e
		}
	}()

	if err := binary.Write(f, binary.LittleEndian, int32(c.dm)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, int32(c.vocabSize)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, int32(c.nLayers)); err != nil {
		return err
	}

	if err = c.model.write(f); err != nil {
		return
	}
	return
}

func (c *converter) convOutLayerNorm() (err error) {
	c.model.ln, err = c.convLayerNorm("ln_out", c.params)
	if err != nil {
		err = fmt.Errorf("failed to convert layer-norm: %w", err)
	}
	return
}

func (c *converter) convRWKVLayerNorm() (err error) {
	c.model.rwkv.ln, err = c.convLayerNorm("blocks.0.ln0", c.params)
	if err != nil {
		err = fmt.Errorf("failed to convert layer-norm 0: %w", err)
	}
	return
}

func (c *converter) convEmbeddings() error {
	embWeight, err := c.params.get("emb.weight")
	if err != nil {
		return err
	}

	matrix, err := c.tensorToVectors(embWeight)
	if err != nil {
		return fmt.Errorf("failed to convert embeddings: %w", err)
	}

	c.vocabSize = len(matrix)
	c.dm = len(matrix[0])

	flattenedSlice := flattenTransformedEmbeddingMatrix(transformEmbeddingMatrix(matrix))

	c.model.emb = array2D{
		rows: c.dm,
		cols: c.vocabSize,
		data: flattenedSlice,
	}
	return nil
}

func transformEmbeddingMatrix(matrix [][]float32) [][]float32 {
	dimSize := len(matrix[0])
	vocabSize := len(matrix)

	transformedMatrix := make([][]float32, dimSize)
	for i := 0; i < dimSize; i++ {
		transformedMatrix[i] = make([]float32, vocabSize)
		for j := 0; j < vocabSize; j++ {
			transformedMatrix[i][j] = matrix[j][i]
		}
	}

	return transformedMatrix
}

func flattenTransformedEmbeddingMatrix(matrix [][]float32) []float32 {
	vocabSize := len(matrix)
	dimSize := len(matrix[0])

	flatSlice := make([]float32, vocabSize*dimSize)

	for i := 0; i < vocabSize; i++ {
		for j := 0; j < dimSize; j++ {
			flatSlice[i*dimSize+j] = matrix[i][j]
		}
	}

	return flatSlice
}

func (c *converter) convLinear() error {
	headWeight, err := c.params.get("head.weight")
	if err != nil {
		return err
	}

	m, err := c.tensorToMatrix(headWeight)
	if err != nil {
		return fmt.Errorf("failed to convert head-weight/linear: %w", err)
	}

	if vs := c.vocabSize; m.rows != vs {
		return fmt.Errorf("expected head-weight/linear rows to match vocabulary size %d, actual %d", vs, m.rows)
	}
	if dm := c.dm; m.cols != dm {
		return fmt.Errorf("expected head-weight/linear columns to match DModel %d, actual %d", dm, m.cols)
	}

	c.model.proj = m
	return nil
}

func (c *converter) convBlocks() error {
	allBlocksParams := c.params.fetchPrefixed("blocks.")
	numBlocks, err := countBlocks(allBlocksParams)
	if err != nil {
		return err
	}
	if numBlocks == 0 {
		return fmt.Errorf("no blocks/layers found in parameters")
	}
	c.nLayers = numBlocks

	layers := make([]rwkvLayerType, numBlocks)
	for i := range layers {
		blockParams := allBlocksParams.fetchPrefixed(fmt.Sprintf("%d.", i))
		layers[i], err = c.convBlock(i, blockParams)
		if err != nil {
			return fmt.Errorf("failed to convert block/layer %d: %w", i, err)
		}
	}

	c.model.rwkv.layers = layers
	return nil
}

func (c *converter) convBlock(id int, params paramsMap) (_ rwkvLayerType, err error) {
	layer := rwkvLayerType{}

	layer.channelMix, err = c.convChanMix(id, params.fetchPrefixed("ffn."))
	if err != nil {
		return rwkvLayerType{}, fmt.Errorf("failed to convert ffn/channel-mix: %w", err)
	}

	layer.timeMix, err = c.convTimeMix(id, params.fetchPrefixed("att."))
	if err != nil {
		return rwkvLayerType{}, fmt.Errorf("failed to convert att/time-mix: %w", err)
	}

	layer.ln1, err = c.convLayerNorm("ln1", params)
	if err != nil {
		return rwkvLayerType{}, fmt.Errorf("failed to convert layer-norm 1: %w", err)
	}

	layer.ln2, err = c.convLayerNorm("ln2", params)
	if err != nil {
		return rwkvLayerType{}, fmt.Errorf("failed to convert layer-norm 2: %w", err)
	}

	return layer, nil
}

func (c *converter) convChanMix(id int, params paramsMap) (channelMixType, error) {
	dm := c.dm

	wk, err := c.fetchParamToMatrix(params, "key.weight", [2]int{dm * 4, dm})
	if err != nil {
		return channelMixType{}, fmt.Errorf("failed to convert wk weight: %w", err)
	}

	wr, err := c.fetchParamToMatrix(params, "receptance.weight", [2]int{dm, dm})
	if err != nil {
		return channelMixType{}, fmt.Errorf("failed to convert wr weight: %w", err)
	}

	wv, err := c.fetchParamToMatrix(params, "value.weight", [2]int{dm, dm * 4})
	if err != nil {
		return channelMixType{}, fmt.Errorf("failed to convert wv weight: %w", err)
	}

	tmk, err := c.fetchParamToSqueezedVector(params, "time_mix_k", dm)
	if err != nil {
		return channelMixType{}, fmt.Errorf("failed to convert time-mix-k: %w", err)
	}

	tmr, err := c.fetchParamToSqueezedVector(params, "time_mix_r", dm)
	if err != nil {
		return channelMixType{}, fmt.Errorf("failed to convert time-mix-r: %w", err)
	}

	return channelMixType{
		wk:   wk,
		wv:   wv,
		wr:   wr,
		mixK: tmk,
		mixR: tmr,
	}, nil
}

func (c *converter) convTimeMix(id int, params paramsMap) (timeMixType, error) {
	dm := c.dm

	wk, err := c.fetchParamToMatrix(params, "key.weight", [2]int{dm, dm})
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert key weight: %w", err)
	}

	wr, err := c.fetchParamToMatrix(params, "receptance.weight", [2]int{dm, dm})
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert receptance weight: %w", err)
	}

	wo, err := c.fetchParamToMatrix(params, "output.weight", [2]int{dm, dm})
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert output weight: %w", err)
	}

	wv, err := c.fetchParamToMatrix(params, "value.weight", [2]int{dm, dm})
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert value weight: %w", err)
	}

	tDecay, err := c.fetchParamToSqueezedVector(params, "time_decay", dm)
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert time-decay: %w", err)
	}
	for i := range tDecay.data {
		tDecay.data[i] = float32(math.Exp(float64(tDecay.data[i]))) * -1
	}

	tFirst, err := c.fetchParamToSqueezedVector(params, "time_first", dm)
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert time-first: %w", err)
	}

	tmk, err := c.fetchParamToSqueezedVector(params, "time_mix_k", dm)
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert time-mix-k: %w", err)
	}

	tmr, err := c.fetchParamToSqueezedVector(params, "time_mix_r", dm)
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert time-mix-r: %w", err)
	}

	tmv, err := c.fetchParamToSqueezedVector(params, "time_mix_v", dm)
	if err != nil {
		return timeMixType{}, fmt.Errorf("failed to convert time-mix-v: %w", err)
	}

	return timeMixType{
		wk:        wk,
		wv:        wv,
		wr:        wr,
		wo:        wo,
		timeDecay: tDecay,
		timeFirst: tFirst,
		mixK:      tmk,
		mixV:      tmv,
		mixR:      tmr,
	}, nil
}

func (c *converter) convLayerNorm(name string, params paramsMap) (layerNormType, error) {
	dm := c.dm

	w, err := c.fetchParamToVector(params, name+".weight", dm)
	if err != nil {
		return layerNormType{}, fmt.Errorf("failed to convert layer-norm weight: %w", err)
	}

	b, err := c.fetchParamToVector(params, name+".bias", dm)
	if err != nil {
		return layerNormType{}, fmt.Errorf("failed to convert layer-norm bias: %w", err)
	}

	return layerNormType{
		g: w,
		b: b,
	}, nil
}

func (c *converter) loadTorchModelParams() error {
	torchModel, err := pytorch.Load(c.inFilename)
	if err != nil {
		return fmt.Errorf("failed to load torch model %q: %w", c.inFilename, err)
	}
	c.params, err = makeParamsMap(torchModel)
	for k, v := range c.params {
		log.Printf("param %q: %v", k, v.Size)
	}
	if err != nil {
		return fmt.Errorf("failed to read model params: %w", err)
	}
	return nil
}

func (c *converter) tensorToVectors(t *pytorch.Tensor) ([][]float32, error) {
	if len(t.Size) != 2 {
		return nil, fmt.Errorf("expected 2 dimensions, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return nil, err
	}

	rows := t.Size[0]
	cols := t.Size[1]

	vecs := make([][]float32, rows)
	for i := range vecs {
		d := data[i*cols : (i*cols)+cols]
		vecs[i] = d
	}

	return vecs, nil
}

func (c *converter) tensorToMatrix(t *pytorch.Tensor) (array2D, error) {
	if len(t.Size) != 2 {
		return array2D{}, fmt.Errorf("expected 2 dimensions, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return array2D{}, err
	}

	return array2D{
		rows: t.Size[0],
		cols: t.Size[1],
		data: data,
	}, nil
}

func (c *converter) tensorToVector(t *pytorch.Tensor) (array1D, error) {
	if len(t.Size) != 1 {
		return array1D{}, fmt.Errorf("expected 1 dimension, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return array1D{}, err
	}

	return array1D{data: data}, nil
}

func (c *converter) tensorToSqueezedVector(t *pytorch.Tensor) (array1D, error) {
	data, err := c.tensorData(t)
	if err != nil {
		return array1D{}, err
	}
	return array1D{data: data}, nil
}

func (c *converter) tensorData(t *pytorch.Tensor) ([]float32, error) {
	st, ok := t.Source.(*pytorch.BFloat16Storage)
	if !ok {
		return nil, fmt.Errorf("only BFloat16Storage is supported, actual %T", t.Source)
	}
	size := tensorDataSize(t)
	return st.Data[t.StorageOffset : t.StorageOffset+size], nil
}

func (c *converter) fetchParamToVector(params paramsMap, name string, expectedSize int) (array1D, error) {
	t, err := params.get(name)
	if err != nil {
		return array1D{}, err
	}
	v, err := c.tensorToVector(t)
	if err != nil {
		return array1D{}, err
	}
	if len(v.data) != expectedSize {
		return array1D{}, fmt.Errorf("expected vector size %d, actual %d", expectedSize, len(v.data))
	}
	return v, nil
}

func (c *converter) fetchParamToSqueezedVector(params paramsMap, name string, expectedSize int) (array1D, error) {
	t, err := params.get(name)
	if err != nil {
		return array1D{}, err
	}
	v, err := c.tensorToSqueezedVector(t)
	if err != nil {
		return array1D{}, err
	}
	if len(v.data) != expectedSize {
		return array1D{}, fmt.Errorf("expected squeezed vector size %d, actual %d", expectedSize, len(v.data))
	}
	return v, nil
}

func (c *converter) fetchParamToMatrix(params paramsMap, name string, expectedSize [2]int) (array2D, error) {
	t, err := params.get(name)
	if err != nil {
		return array2D{}, err
	}
	m, err := c.tensorToMatrix(t)
	if err != nil {
		return array2D{}, err
	}
	if m.rows != expectedSize[0] || m.cols != expectedSize[1] {
		return array2D{}, fmt.Errorf("expected matrix size %dx%d, actual %dx%d",
			expectedSize[0], expectedSize[1], m.rows, m.cols)
	}
	return m, nil
}

func countBlocks(params paramsMap) (int, error) {
	blocks := 0
	for k := range params {
		before, _, ok := strings.Cut(k, ".")
		if !ok {
			return 0, fmt.Errorf("block/layer parameter names expected to start with number, actual name %q", k)
		}
		num, err := strconv.Atoi(before)
		if err != nil {
			return 0, fmt.Errorf("block/layer parameter names expected to start with number, actual name %q: %w", k, err)
		}
		if num > blocks {
			blocks = num
		}
	}
	return blocks + 1, nil
}

func tensorDataSize(t *pytorch.Tensor) int {
	size := t.Size[0]
	for _, v := range t.Size[1:] {
		size *= v
	}
	return size
}

func cast[T any](v any) (t T, _ error) {
	t, ok := v.(T)
	if !ok {
		return t, fmt.Errorf("type assertion failed: expected %T, actual %T", t, v)
	}
	return
}

type paramsMap map[string]*pytorch.Tensor

func makeParamsMap(torchModel any) (paramsMap, error) {
	od, err := cast[*types.OrderedDict](torchModel)
	if err != nil {
		return nil, err
	}

	params := make(paramsMap, od.Len())

	for k, item := range od.Map {
		name, err := cast[string](k)
		if err != nil {
			return nil, fmt.Errorf("wrong param name type: %w", err)
		}
		tensor, err := cast[*pytorch.Tensor](item.Value)
		if err != nil {
			return nil, fmt.Errorf("wrong value type for param %q: %w", name, err)
		}
		params[name] = tensor
	}

	return params, nil
}

func (p paramsMap) get(name string) (*pytorch.Tensor, error) {
	t, ok := p[name]
	if !ok {
		return nil, fmt.Errorf("parameter %q not found", name)
	}
	delete(p, name)
	return t, nil
}

func (p paramsMap) fetchPrefixed(prefix string) paramsMap {
	out := make(paramsMap, len(p))
	for k, v := range p {
		if after, ok := strings.CutPrefix(k, prefix); ok {
			out[after] = v
			delete(p, k)
		}
	}
	return out
}
