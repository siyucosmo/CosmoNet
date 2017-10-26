Input = {
        "BATCH_SIZE" : 40,
        "NUM_THREADS" : 2,
        "CAPACITY" : 0,
        "MIN_AFTER_DEQUEUE" : 400
        }

Input["CAPACITY"] = Input["BATCH_SIZE"]*4 + Input["MIN_AFTER_DEQUEUE"]

Input_Test = {
	"BATCH_SIZE" : 64,
	"NUM_THREADS" : 2,
	"CAPACITY" : 0,
	"MIN_AFTER_DEQUEUE" : 64
	}

Input_Test["CAPACITY"] = Input_Test["BATCH_SIZE"]*4 + Input_Test["MIN_AFTER_DEQUEUE"]

Model = {
        "REG_RATE": 0.,
        "LEAK_PARAMETER": 0.01,
        "LEARNING_RATE" : 0.0001,
        "DROP_OUT": 0.5
}

RUNPARAM={
	"num_epoch": 1000,
	"num_train":400,
	"num_val":50,
	"batch_per_epoch":0,
	"batch_per_epoch_val":0,
        "iter_test":49
}

RUNPARAM["batch_per_epoch"] = RUNPARAM['num_train']*64/Input['BATCH_SIZE']
RUNPARAM["batch_per_epoch_val"] = RUNPARAM['num_val']*64/Input['BATCH_SIZE']
#RUNPARAM["batch_per_epoch"] = 100
#RUNPARAM["batch_per_epoch_val"] = 500

