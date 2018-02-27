import Model.modelManager as modelManager
import Utilities
import tensorflow as tf

model = modelManager.CreateModel((3,96,96))

print("Total Params:", model.count_params())

model.compile(optimizer = 'adam', loss = modelManager.CalculateTripletLoss, metrics = ['accuracy'])

modelManager.loadWeights(model)

collection = {}
collection["aubrey"] = Utilities.GetImageData("images/aubrey_r.png", model)
collection["cara"] = Utilities.GetImageData("images/cara_r.png", model)
collection["eva"] = Utilities.GetImageData("images/eva_r.png", model)
collection["gal"] = Utilities.GetImageData("images/gal_r.png", model)
collection["keanu"] = Utilities.GetImageData("images/keanu_r.png", model)
collection["margot"] = Utilities.GetImageData("images/margot_r.png", model)
collection["rachel"] = Utilities.GetImageData("images/rachel_r.png", model)
collection["tom"] = Utilities.GetImageData("images/tom_r.png", model)
#collection["tom_hanks"] = Utilities.GetImageData("images/tom_h_r.png", model)
collection["tom_hanks"] = Utilities.GetImageData("images/tom_hanks1_r.png", model)
collection["tom_hardy"] = Utilities.GetImageData("images/tom_ha_r.png", model)



print('end')