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

---------------------------------------------------------------------------------------------------

250612 첫번째 시도
모델은 동일하며 LSTM 모델의 depth 추가(자세한 내용은 주제1번_(LSTM 최종모델) 코드 참조)
확인된 RMSE값: 20.8


250612 두번째 시도
다음와 같이 모델링 진행
 model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, len(feature_cols))),
        Dropout(0.1),
        LSTM(128, return_sequences=False),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
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
확인된 RMSE값: 21.46

---------------------------------------------------------------------------------------------------

진의형 try: 각 연도별 값 평균을 통해 데이터 변환 후 모델링은 그대로 사용
확인된 RMSE값: 19?(확인 부탁드려요!)

---------------------------------------------------------------------------------------------------

250616 첫번째 시도
KerasTuner를 사용해 지점별로 최적의 모델 파라미터 값 튜닝 진행(Bayesian Optimizer 사용)
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('lstm_units', min_value=64, max_value=256, step=64),
        return_sequences=True,
        input_shape=(seq_len, len(feature_cols))
    ))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    model.add(LSTM(
        units=hp.Int('lstm_units2', min_value=32, max_value=128, step=32),
        return_sequences=False
    ))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))

    model.add(Dense(hp.Int('dense1', 32, 128, step=32), activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 5e-4])),
        loss='mse'
    )
    return model
가장 Best 성능을 뽑아내는(Score가 낮은) 모델을 사용해 test 데이터 예측
확인된 RMSE값: 22.04

