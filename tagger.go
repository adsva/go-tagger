package main

import (
	"bufio"
	"container/list"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

func ReadSentences() (sentences chan *list.List, err error) {

	sentences = make(chan *list.List)

	go func() {
		defer close(sentences)

		reader := bufio.NewReader(os.Stdin)
		sentence := list.New()
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					fmt.Println("Error reading tokens from stdin:", err)
				}
				sentences <- sentence
				return
			}

			token := strings.Trim(line, "\n")
			if token != "" {
				sentence.PushBack(token)
			}
			if len(token) == 1 && strings.IndexAny(token, ".?!:") == 0 {
				sentences <- sentence
				sentence = list.New()
			}
		}
	}()

	return
}

func main() {
	trainPath := flag.String("train", "", "Path to training data")
	window := flag.Int("window", 3, "Context window width")

	flag.Parse()

	if *trainPath == "" {
		fmt.Print("\nYou will have to at least supply a path to the training data.\nThe input to tag will be read from stdin and printed to stdout.\n")
		os.Exit(1)
	}

	posModel := NewHMM("POS", *window)

	instances, errors := ReadTrainingData(*trainPath)

	if err := posModel.Train(instances, errors); err != nil {
		fmt.Printf("Training error: %v\n", err)
		os.Exit(1)
	}
	//posModel.PrintModel()
	//os.Exit(0)

	// tokenizer, err := NewTokenizer()
	// if err != nil {
	// 	fmt.Printf("Tokenizer error: %v\n", err)
	// 	os.Exit(1)
	// }

	// tokens, errors, err := tokenizer.TokenizeFile(*tagPath)
	// if err != nil {
	// 	fmt.Printf("Tokenizing error: %v\n", err)
	// 	os.Exit(1)
	// }

	sentences, err := ReadSentences()
	if err != nil {
		fmt.Println("Error reading input tokens:", err)
		os.Exit(1)
	}

	var allprocessedtokens int
	for sentence := range sentences {
		tags, processedtokens, err := posModel.TagViterbi(sentence)
		if err != nil {
			fmt.Printf("Testing error: %v\n", err)
			os.Exit(1)
		}

		allprocessedtokens += processedtokens
		token, tag := sentence.Front(), tags.Front()
		for token != nil {
			fmt.Printf("%s\t%s\n", token.Value, tag.Value)
			token, tag = token.Next(), tag.Next()
		}
	}

	fmt.Fprintf(os.Stderr, "Number of processed tokens: %v\n", allprocessedtokens)
	return
}
