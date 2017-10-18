Input = {
        "BATCH_SIZE" : 160,
        "NUM_THREADS" : 4,
        "CAPACITY" : 0,
        "MIN_AFTER_DEQUEUE" : 80
        }

Input["CAPACITY"] = Input["BATCH_SIZE"]*4 + Input["MIN_AFTER_DEQUEUE"]

Model = {
        "REG_RATE": 0.,
        "LEAK_PARAMETER": 0.01,
        "LEARNING_RATE" : 0.0001,
        "DROP_OUT": 0.5
}

RUNPARAM={
	"num_epoch": 200,
	"num_train":450,
	"num_val":45,
	"batch_per_epoch":0,
	"batch_per_epoch_val":0
}

RUNPARAM["batch_per_epoch"] = RUNPARAM['num_train']*64/Input['BATCH_SIZE']
RUNPARAM["batch_per_epoch_val"] = RUNPARAM['num_val']*64/Input['BATCH_SIZE']
#RUNPARAM["batch_per_epoch"] = 100
#RUNPARAM["batch_per_epoch_val"] = 500

