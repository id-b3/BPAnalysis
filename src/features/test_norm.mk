SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all clean

comma:=,
P_LIST := $(subst $(comma), , $(PARAMS))
# CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: hist

hist: $(BP_FINAL)
	# $(CONDA_ACTIVATE) stats
	for col in $(P_LIST); do \
		echo $$col
		csvcut -c "$$col,gender,GOLD_stage,copd_diagnosis,asthma_diagnosis" $< | python ./src/features/descriptive/test_normality.py ; \
	done
