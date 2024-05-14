from multiprocessing import Process, freeze_support
import os
from agent import CreateModel, MakeEnsembleModel, MakeModel, PlayWithTraining, PlayWithTrainingEnsemble


def main():
    model = CreateModel((13,), 4)
    # save in keras format
    training_processes = []
    for i in ["tiny","small","med","large"]:
        snakeLength = 3 if i == "tiny" else 5 if i == "small" else 7 if i == "med" else 9
        training_processes.append(Process(target=MakeModel, args=(i, snakeLength)))
        training_processes[-1].start()
    for process in training_processes:
        process.join()
        
    # models = []
    # for i in ["tiny", "small","med","large"]:
    #     model = CreateModel((14,), 4)
    #     fileName = len(os.listdir(f'./submodels/{i}')) - 1
    #     path = f"./submodels/{i}/{fileName}.h5"
    #     model.load_weights(path)
    #     models.append(model)
    # ensemble = MakeEnsembleModel(models)

    # PlayWithTraining(models[0])

    # ensemble = CreateModel((5,), 4)
    # ensemble.load_weights("model_snake7.h5")
    # ensemble.save("model_snake7.h5")
    # PlayWithTrainingEnsemble(ensemble, models)



if __name__ == "__main__":
    freeze_support()
    main()


