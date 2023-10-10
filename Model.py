def create_model_Densenet(model_name):
  models = { "Densenet" : Densenet }
  model = models[model_name]
  for layer in model.layers:
    layer.trainable = True
  # x = tf.keras.layers.Flatten()(model.output)
  x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
  x= tf.keras.layers.Dense(5,activation='softmax')(x)
  model = Model(inputs= model.input, outputs=x)
  my_model = tf.keras.models.clone_model(model)
  return my_model 
# model = tf.keras.Model(feature_extractor_model.input, x)
model=create_model_Densenet('Densenet')


model.compile( optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history=Densenet.fit(x = train_data,epochs=200,validation_data=val_data,verbose=1,callbacks=my_callbacks,shuffle=True)

