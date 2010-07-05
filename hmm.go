package main

import (
	"fmt"
	"os"
	"bufio"
	"strings"
	"container/vector"
	"container/list"
	"math"
)

const TAGSEP string = "|"

type Stats struct {
	freq float
	prob float
}

func (stats *Stats) String() string {
	return fmt.Sprintf("%+v", *stats)
}

type TagMap map[string]*Stats 

type Suffix struct {
	count float
	tags TagMap
}
func NewSuffix() *Suffix {
	return &Suffix{0.0, make(TagMap)}
}

type SuffixMap map[string]*Suffix

// Lexicon["can"]["NN1"].prob -> p(can|NN1)
// Ngrams[2]["AT0:NN1"].prob -> p(NN1|AT0)
type HMM struct {
	Name             string
	Window           int
	Lexicon          map[string]TagMap 
	Ngrams           map[int]TagMap
	Suffixes         map[int]SuffixMap
	InstanceCount    float
	NgramWeights     map[int]float
	SuffixWeights    map[int]float
	MaxSuffixLen     int
	MostFreqTag      string
}

func (model *HMM) PrintModel() {
	fmt.Println("BEGIN LEXICON")
	for word, tags := range model.Lexicon {
		fmt.Print(word)
		for tag, stats := range tags {
			fmt.Printf("\t%s:%s", tag, stats)	
		}
		fmt.Print("\n")
	}	
	fmt.Println("END LEXICON")
	fmt.Println("BEGIN SUFFIXES")
	for _, suffixes := range model.Suffixes {
		for suffix, suffixData := range suffixes {
			fmt.Print(suffix)
			for tag, stats := range suffixData.tags {
				fmt.Printf("\t%s:%s", tag, stats)	
			}
			fmt.Print("\n")
		}
	}
	fmt.Println("END SUFFIXES")
	fmt.Print("SUFFIX WEIGHTS:")
	for length, weight := range model.SuffixWeights {
		fmt.Printf(" %d:%.20f", length, weight)
	}
	fmt.Print("\n")
}
	
func NewHMM(name string, window int) *HMM {
	model := new(HMM)
	model.Name = name
	model.Window = window
	model.Lexicon = make(map[string]TagMap)
	model.Ngrams = make(map[int]TagMap)
	model.NgramWeights = make(map[int]float)
	model.Suffixes = make(map[int]SuffixMap)
	model.SuffixWeights = make(map[int]float)
	for n := 1; n <= window; n++ {
		model.Ngrams[n] = make(TagMap)
		model.NgramWeights[n] = 0.0
	}
	return model
}

// Calculate smoothing weights by deleted interpolation 
func (model *HMM) calcNgramWeights() (err os.Error) {
	var lowOrderFreq, highOrderFreq, highOrderProb float
	var	maxN int
	max := new(Stats)

	for ngram, _ := range model.Ngrams[model.Window] {
		for n := model.Window; n > 0; n-- {
			highOrderFreq = model.Ngrams[n][ngram].freq

			if n > 1 {
				// T1:T2:T3 -> T2:T3
				idx := strings.Index(ngram, TAGSEP) 
				ngram = ngram[idx+1:] 
				lowOrderFreq = model.Ngrams[n-1][ngram].freq
			} else {
				lowOrderFreq = model.InstanceCount
			}
			if lowOrderFreq-1 < 1.0 {
				highOrderProb = 0.0
			} else {
				highOrderProb = (highOrderFreq -1) / (lowOrderFreq - 1)
			}
			if highOrderProb > max.prob {
				max.prob = highOrderProb
				max.freq = highOrderFreq
				maxN = n
			}
		}
		model.NgramWeights[maxN] += max.freq
	}
	// Normalize weights
	weightSum := 0.0
	for _, weight := range model.NgramWeights {
		weightSum += weight
	}
	for n, weight := range model.NgramWeights {
		model.NgramWeights[n] = weight / weightSum
	}

	return
}

// Calculate probabilities.
// The probabilities are smoothed and stored for each ngram, even
// though only the probability for the longest ngram (n=window) is
// used in tagging. This is because the probability of an unseen
// ngram then can be accessed as the probability of the first found
// lower-order ngram.
func (model *HMM) calcProbs() (err os.Error) {

	// Store per suffix length tag counts for weight calculations
	suffixLenTagCounts := make(map[int]map[string]float)
	suffixLenTagSums := make(map[int]float)

	// Lexical : p(w|t) = f(w,t)/f(t) 
	for token, tags := range model.Lexicon {
		tokenFreq := 0.0
		for tag, tStats := range tags {
			tokenTagFreq := model.Ngrams[1][tag].freq
			tStats.prob = tStats.freq / tokenTagFreq
			tokenFreq += tokenTagFreq
		}

		// Build suffix model using only uncommon words
		if tokenFreq > 100 {
			continue
		}
			
		// Use at most 1/3 of the word as suffix
		tokenLen := len(token)
		for cutoff := int(float(tokenLen)*0.6); cutoff < tokenLen; cutoff++ {
			suffix := token[cutoff:]
			suffixLen := len(suffix)

			suffixes, ok := model.Suffixes[suffixLen]
			if !ok {
				suffixes = make(SuffixMap)
				model.Suffixes[suffixLen] = suffixes
				suffixLenTagCounts[suffixLen] = make(map[string]float)
				suffixLenTagSums[suffixLen] = 0.0
			}

			suffixData, ok := suffixes[suffix]
			if !ok {
				suffixData = new(Suffix)
				suffixData.tags = make(TagMap)
				suffixes[suffix] = suffixData
			}
			
			tagCounts := suffixLenTagCounts[suffixLen]
			for tag, tagStats := range tags {
				stats, ok := suffixData.tags[tag]
				if !ok {
					stats = new(Stats)
					suffixData.tags[tag] = stats
				}
				stats.freq += tagStats.freq
				suffixData.count += tagStats.freq
				
				if _, ok := tagCounts[tag]; !ok {
					tagCounts[tag] = tagStats.freq
				} else {
					tagCounts[tag] += tagStats.freq
				}
				suffixLenTagSums[suffixLen] += tagStats.freq
			}
		}
	}

	// Unigram : p(t) = unigram weight * (f(t)/N) 
	uniWeight := model.NgramWeights[1]
	maxTag, maxTagFreq  := "EMTPY", 0.0

	for tag, tagStats := range model.Ngrams[1] {
		tagStats.prob = uniWeight * (tagStats.freq / model.InstanceCount)
		if tagStats.freq > maxTagFreq {
			maxTag, maxTagFreq = tag, tagStats.freq
		}
	}
	model.MostFreqTag = maxTag

	// Ngram (n > 1) : p(t2|t1) = smoothP(t2|t1) + (weight * f(t1,t2)/f(t1))
	for n := 2; n <= model.Window; n++ {
		nWeight := model.NgramWeights[n]
		for ngram, nstats := range model.Ngrams[n] {
			context := ngram[0:strings.LastIndex(ngram, TAGSEP)]
			nProb := nWeight * (nstats.freq / model.Ngrams[n-1][context].freq)
			nstats.prob = nProb + model.Ngrams[n-1][context].prob
		}
	}


	// Suffix probs
	model.MaxSuffixLen = len(model.Suffixes)
	for suffixLen := 1; suffixLen <= len(model.Suffixes); suffixLen++ { 

		// Suffix weights : standard deviation of tag likelihoods for
		// each suffix length. I guess the idea is that a higher
		// standard dev means that suffix length set contains more
		// tag-specific suffixes?
		
		tagSum := suffixLenTagSums[suffixLen]
		numTags := len(suffixLenTagSums)
		tagProbs := new(vector.Vector)
		tagProbSum := 0.0
		for _, tagCount := range suffixLenTagCounts[suffixLen] {
			tagProb := tagCount/tagSum
			tagProbs.Push(tagProb)
			tagProbSum += tagProb
		}
		tagProbAvg := tagProbSum/float(tagProbs.Len())
		tagVariance := 0.0
		for tagProb := range tagProbs.Iter() {
			tagVariance += float(math.Pow(float64(tagProb.(float) - tagProbAvg), 2))
		}
		suffixLenWeight := (1.0/(float(numTags)-1.0)) * tagVariance
		model.SuffixWeights[suffixLen] = suffixLenWeight

		// Suffix probs : P(tag|suffix) We're going from short->long,
		// so we can smooth with the previous suffix length by simply
		// adding its pre-smoothed prob
		suffixes := model.Suffixes[suffixLen] 
		for suffix, suffixData := range suffixes {
			for tag, tagStats := range suffixData.tags {
				if suffixLen == 1 {
					// Initialize using (C(tag,suff)/C(suff)) * P(tag)
					tagStats.prob = suffixLenWeight * (tagStats.freq/suffixData.count) * model.Ngrams[1][tag].prob
				} else {
					prevSuffix := suffix[1:]
					prevProb := model.Suffixes[suffixLen-1][prevSuffix].tags[tag].prob // Smoothing prob
					prevWeight := model.SuffixWeights[suffixLen-1]     // Needed to normalize smoothed prob 
					tagStats.prob = ((tagStats.freq/suffixData.count) * prevProb) / (1 + prevWeight)
				}
			}
		}
	}
	
	// Now that the probabilities have been smoothed, we use
	// bayesian inversion to transform P(tag|suffix) ->
	// P(suffix|tag) wich can be used instead of P(word|tag) for
	// unkown words. TODO: Try to work this into the previous loop to
	// save iterations.
	for _, suffixes := range model.Suffixes {
		totalSuffixes := len(suffixes)
		for _, suffixData := range suffixes {
			for tag, tagStats := range suffixData.tags {
				// p(suff|tag) = (p(tag|suff) * p(suff)) / p(tag)
				tagStats.prob = (tagStats.prob * (suffixData.count/float(totalSuffixes))) / model.Ngrams[1][tag].prob
			}
		}
	}

	return
}

func (model *HMM) addInstance(instance TrainingInstance) {

	instanceTags, ok := model.Lexicon[instance.token]
	if !ok {
		instanceTags = make(map[string]*Stats)
		model.Lexicon[instance.token] = instanceTags
	}

	tagStats, ok := instanceTags[instance.tag]
	if !ok {
		tagStats = new(Stats)
		instanceTags[instance.tag] = tagStats
	}
	tagStats.freq++

	model.InstanceCount++
	return
}

func (model *HMM) addNgram(tagStr string, n int) {

	tag, ok := model.Ngrams[n][tagStr]
	if !ok {
		tag = new(Stats)
		model.Ngrams[n][tagStr] = tag
	}
	tag.freq++

	return
}

type TrainingInstance struct {
	token string
	tag string
}

func ReadTrainingData(path string) (instances chan TrainingInstance, err chan os.Error) {

	instances, errors := make(chan TrainingInstance), make(chan os.Error)
	go func() {
		defer close(instances)
		defer close(errors)

		file, err := os.Open(path, os.O_RDONLY, 0)
		if err != nil {
			errors <- err
			return
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		lineCount := 0
		for {
			line, err := reader.ReadString('\n')
			lineCount++
			if err != nil {
				if err != os.EOF {
					errors <- err
				}
				return
			}
			cols := strings.Fields(line)
			if len(cols) != 2 {
				fmt.Printf("Skipping malformed line %d:%s\n", lineCount, line)
				continue
			}
			instance := TrainingInstance{token:cols[0], tag:cols[1]}
			instances <- instance
		}
	}()

	return instances, errors
	
}


// Collect frequencies, Calculate ngram probability weights and final probabilites
func (model *HMM) Train(instances chan TrainingInstance, errors chan os.Error) (err os.Error) {

	// Add sequence start tag stats 
	context := make([]string, model.Window)
	for i := 0; i < model.Window; i++ {
		context[i] = "BEG"
	}

	// Process the start tags for ngram stats
	for i := 0; i < model.Window; i++ {
		ngram := strings.Join(context[i:model.Window], TAGSEP)
		model.addNgram(ngram, model.Window-i)
	}

	// Collect training frequencies 
	for instance := range instances {
		model.addInstance(instance)
		
		// Shift context array one step back
		for i, tag := range context {
			if i > 0 {
				context[i-1] = tag
			}
		}
		context[model.Window-1] = instance.tag
		
		// Add tag ngram stats for each n
		for i := 0; i < model.Window; i++ {
			ngram := strings.Join(context[i:model.Window], TAGSEP)
			model.addNgram(ngram, model.Window-i)
		}
	}
	
	if err = <-errors; err != nil {
		return err
	}
	if err := model.calcNgramWeights(); err != nil {
		return err
	}

	if err := model.calcProbs(); err != nil {
		return err
	}

	return
}

func (model *HMM) findTags(token string) (tags map[string]*Stats) {
	if tags, ok := model.Lexicon[token]; ok { 
		return tags 
	}

	// Token not in lexicon. Do backoff suffix matching.
	var cutoff int
	tokenLen := len(token)
	if model.MaxSuffixLen > tokenLen {
		cutoff = 0
	} else {
		cutoff = tokenLen-model.MaxSuffixLen
	}
	for ; cutoff < tokenLen; cutoff++ {
		suffix := token[cutoff:]
		if s, ok := model.Suffixes[len(suffix)][suffix]; ok { 
			return s.tags 
		}
	}
	maxTagMap := make(map[string]*Stats)
	maxTagMap[model.MostFreqTag] = &Stats{1.0, 1.0}
	return maxTagMap
}

type State struct {
	tag     string
	context []string
	prob    float
	prev    *State
}
func (state *State) String() string {
	if state != nil {
		return fmt.Sprintf("%+v", *state)
	} 
	return ""

}

func (model *HMM) TagViterbi(tokens *list.List) (tags *list.List, err os.Error) {

	// Initialize initial state
	states := make(map[string]*State)
	contextSize := model.Window-1
	context := make([]string, contextSize)
	for i, _ := range context {
		context[i] = "EMTPY"
	}
	states["EMPTY"] = &State{"EMPTY", context, 1, nil}

	// Add token states
	for token := range tokens.Iter() {
		token := token.(string)
		tokenTags := model.findTags(token)
		nextStates := make(map[string]*State)
		for _, state := range states {
			for tag, stats := range tokenTags {
				// Build context of next state
				context := make([]string, contextSize)
				copy(context, state.context[1:]) // {Ts-2,Ts-1} -> {Ts-1, nil}
				context[contextSize-1] = state.tag // -> {Ts-1, Ts}
				
				// Transition and emission probabilites, with backoff
				var trProb float
				for i := 0; i < contextSize; i++ {
					ngram := strings.Join(context[i:], TAGSEP) + TAGSEP + tag
					ngramStats, ok  := model.Ngrams[model.Window-i][ngram] // p(Ti|Ti-1..Ti-n)
					if ok {
						trProb = ngramStats.prob
						break
					}
				}
				if trProb == 0.0 {
					trProb = model.Ngrams[1][tag].prob // p(Ti|Ti-1..Ti-n)					
				}

				emProb := stats.prob   // p(w|Ti)
				viterbiProb := state.prob * trProb * emProb

				// If multiple states lead to this state, keep the
				// path with the highest viterbi probability
				nextState, ok := nextStates[tag]
				if !ok || viterbiProb > nextState.prob {
					nextStates[tag] = &State{tag, context, viterbiProb, state}
				} 
			}
		}
		states = nextStates
	}

	// Find best end state (has to be a nicer way. srlsy.)
	var bestState *State
	for _, state := range states {
		if bestState == nil {
			bestState = state
		} else if state.prob >= bestState.prob {
			bestState = state
		}
	} 

	// Backtrack from the best end state
	tags = list.New()
	for bestState.tag != "EMPTY" {
		tags.PushFront(bestState.tag)
		bestState = bestState.prev
	}
	
	return tags, nil
}
