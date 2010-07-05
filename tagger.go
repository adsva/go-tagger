package main

import (
	"os"
	"flag"
	"fmt"
	"bufio"
	"strings"
	"container/list"
)

func ReadSentences(path string) (sentences chan *list.List,  err os.Error ) {

	sentences = make(chan *list.List)

	file, err := os.Open(path, os.O_RDONLY, 0)
	if err != nil {
		return
	}
	
	go func() {
		defer close(sentences)
		defer file.Close()
		
		reader := bufio.NewReader(file)
		sentence := list.New()
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != os.EOF {
					fmt.Println("Error reading tokens from", path, err)
				}
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
	tagPath   := flag.String("tag", "", "Path to tagger input data")
	window    := flag.Int("window", 3, "Context window width")

	flag.Parse()

	if *trainPath == "" || *tagPath == "" {
		fmt.Println("Supply train/tag paths")
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

	sentences, err := ReadSentences(*tagPath)
	if err != nil {
		fmt.Println("Error reading input tokens:", err)
		os.Exit(1)
	}

	for sentence := range sentences {
		tags, err := posModel.TagViterbi(sentence)
		if  err != nil {
			fmt.Printf("Testing error: %v\n", err)
			os.Exit(1)
		}

		token, tag := sentence.Front(), tags.Front()
		for token != nil {
			fmt.Printf("%s\t%s\n", token.Value, tag.Value)
			token, tag = token.Next(), tag.Next()
		}
	}
		
	return
}
