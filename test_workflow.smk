models = ['10k']
model2l = { '10k': 10000}
seeds = [42, 43, 44, 45, 46]

rule all:
    input: expand('ckpts/{model}_{seed}.pt', model=models, seed=seeds)

rule train:
    input:
        train = lambda wildcards: f'kirai_train_code/Canonical/dataset_train_all.{model2l[wildcards.model]}.h5',
        test = lambda wildcards: f'kirai_train_code/Canonical/dataset_test_0.{model2l[wildcards.model]}.h5',
    output:
        'ckpts/{model}_{seed}.pt'
    shell:
        'python -m kirai_pytorch.train '
        '--model {wildcards.model} '
        '--train-h5 {input.train} '
        '--test-h5 {input.test} '
        '--output {output} '
        '--seed {wildcards.seed} '
        '--use-wandb'
