// Command fakevllm serves a fake vLLM API (LLM, embed and rerank on one port)
// for trying the vLLM backend without a GPU.
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/DressingPluto84/Model-Scheduler/internal/vllm/fakevllm"
)

func main() {
	addr := flag.String("addr", ":8001", "listen address")
	cfg := fakevllm.DefaultConfig()
	flag.DurationVar(&cfg.DecodePerToken, "decode", cfg.DecodePerToken, "time per generated token")
	flag.IntVar(&cfg.KVBlocks, "kv-blocks", cfg.KVBlocks, "KV cache blocks reported in /metrics")
	flag.Parse()
	log.Printf("fake vLLM listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, fakevllm.New(cfg).Handler()))
}
