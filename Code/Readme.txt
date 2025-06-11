공통적인 전처리 방식
1. 풍향(wd)의 경우 결측값이 -9.9인 경우가 있으므로 이를 NA로 교체
2. 나머지 데이터의 경우 결측값이 -99로 되어 있으며 이를 NA로 교체
3. 결측치의 경우 선형보간을 통해 결측치 제거

250611 첫번째 시도
<사용한 모델 - LSTM>
model = Sequential([
  LSTM(64, input_shape=(seq_len, len(feature_cols))),
  Dense(1)
])
model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
  X_train, y_train,
  epochs=100,
  batch_size=32,
  validation_data=(X_val, y_val),
  callbacks=[early_stop],
  verbose=1
)

확인된 RMSE값: 21.03
기본적으로 이 모델에서 parameter 값을 변경해가며 코드 수정 진행


250612 두번째 시도
모델은 동일하며 LSTM 모델의 depth 추가(자세한 내용은 주제1번_(LSTM 최종모델) 코드 참조)
확인된 RMSE값: 20.8
