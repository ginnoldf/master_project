# cosine annealing lr scheduling
qsub submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy16.sh
qsub submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy32.sh
qsub submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy64.sh
qsub submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy128.sh
qsub submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy256.sh
qsub submissions/training/atmosphere/opt/cosine_annealing/all.sh

# step lr scheduling
qsub submissions/training/atmosphere/opt/step_lr/lz2_lxy16.sh
qsub submissions/training/atmosphere/opt/step_lr/lz2_lxy32.sh
qsub submissions/training/atmosphere/opt/step_lr/lz2_lxy64.sh
qsub submissions/training/atmosphere/opt/step_lr/lz2_lxy128.sh
qsub submissions/training/atmosphere/opt/step_lr/lz2_lxy256.sh
qsub submissions/training/atmosphere/opt/step_lr/all.sh

# from pretrained
qsub submissions/training/atmosphere/maml/from_pretrained/lz2_lxy16/01.sh
qsub submissions/training/atmosphere/maml/from_pretrained/lz2_lxy16/02.sh
qsub submissions/training/atmosphere/maml/from_pretrained/lz2_lxy256/01.sh

# randomly initialized
qsub submissions/training/atmosphere/maml/randomly_initialized/01.sh
qsub submissions/training/atmosphere/maml/randomly_initialized/02.sh
qsub submissions/training/atmosphere/maml/randomly_initialized/03.sh
qsub submissions/training/atmosphere/maml/randomly_initialized/04.sh

# targeted maml
qsub submissions/training/atmosphere/maml/targeted/lz2_lxy16/01.sh
qsub submissions/training/atmosphere/maml/targeted/lz2_lxy32/01.sh
qsub submissions/training/atmosphere/maml/targeted/lz2_lxy64/01.sh
qsub submissions/training/atmosphere/maml/targeted/lz2_lxy128/01.sh
qsub submissions/training/atmosphere/maml/targeted/lz2_lxy256/01.sh

# transfer learning
qsub submissions/training/atmosphere/transfer_learning/unfreeze_cnn12.sh
qsub submissions/training/atmosphere/transfer_learning/unfreeze_dnn3.sh
qsub submissions/training/atmosphere/transfer_learning/unfreeze_dnn23.sh
