

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
optimizer = dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
