# step lr scheduling
qsub submissions/training/ocean/opt/step_lr/lsel50.sh
qsub submissions/training/ocean/opt/step_lr/lsel100.sh
qsub submissions/training/ocean/opt/step_lr/lsel200.sh
qsub submissions/training/ocean/opt/step_lr/lsel400.sh
qsub submissions/training/ocean/opt/step_lr/all.sh

# cosine annealing lr scheduling
qsub submissions/training/ocean/opt/cosine_annealing/lsel50.sh
qsub submissions/training/ocean/opt/cosine_annealing/lsel100.sh
qsub submissions/training/ocean/opt/cosine_annealing/lsel200.sh
qsub submissions/training/ocean/opt/cosine_annealing/lsel400.sh
qsub submissions/training/ocean/opt/cosine_annealing/all.sh

