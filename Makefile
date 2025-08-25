# Makefile

QWEN_HF_PATH ?= Qwen2.5-0.5B-Instruct
QWEN_BIN_MODEL_PATH ?= Qwen2.5-0.5B-Instruct.bin
QWEN_BIN_TOKEN_PATH ?= tokenizer.bin
CC ?= gcc # or clang

########## Hf Model ###########
${QWEN_HF_PATH}:
	@if [ ! -f "${QWEN_HF_PATH}" ]; then \
		echo "Error: ${QWEN_HF_PATH} does not exist!"; \
		exit 1; \
	fi

########## Export bin Model ###########
${QWEN_BIN_MODEL_PATH}: ${QWEN_HF_PATH}
	python export/export_qwen2_bin.py ${QWEN_BIN_MODEL_PATH} --hf=${QWEN_HF_PATH};
	

${QWEN_BIN_TOKEN_PATH}: ${QWEN_HF_PATH} 
	python export/export_token_bin.py ${QWEN_HF_PATH} ${QWEN_BIN_TOKEN_PATH}

########## Plain C run ###########
all_c_infer: run.c ${QWEN_BIN_MODEL_PATH} ${QWEN_BIN_TOKEN_PATH}
	$(CC) -Ofast -fopenmp -march=native -DNORMAL -DQWEN_BIN_MODEL_PATH='"${QWEN_BIN_MODEL_PATH}"' -DQWEN_BIN_TOKEN_PATH='"${QWEN_BIN_TOKEN_PATH}"' run.c  -lm  -o run

clean:
	rm -f *.so
	rm run
