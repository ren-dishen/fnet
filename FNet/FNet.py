import Model.modelManager as modelManager

model = modelManager.CreateModel((3,96,96))

print("Total Params:", model.count_params())

model.compile(optimizer = 'adam', loss = modelManager.CalculateTripletLoss, metrics = ['accuracy'])



print('end')