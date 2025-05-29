async def binance_ws_listener():
    interval_seconds = interval_map[selected_tf]
    binance_interval_map = {
        300: '5m',
        900: '15m',
        1800: '30m',
        3600: '1h',
        14400: '4h',
        86400: '1d'
    }
    ws_interval = binance_interval_map[interval_seconds]
    
    uri = f"wss://stream.binance.com:9443/ws/xauusdt@kline_{ws_interval}"
    
    async with websockets.connect(uri) as websocket:
        candle_data = []
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            
            # Check if kline is closed
            kline = data['k']
            is_closed = kline['x']
            if is_closed:
                candle = {
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'timestamp': int(kline['t'])
                }
                candle_data.append(candle)
                
                # Convert candle_data to DataFrame
                df = pd.DataFrame(candle_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                
                # Generate signals and plot
                if len(df) > 20:
                    df_signals = generate_signals(df)
                    
                    chart_placeholder.empty()
                    plot_signals(df_signals, f"Gold (XAU/USDT) - Buy/Sell Signals [{selected_tf}]")
                    
                    latest_signal = df_signals[(df_signals['Buy'] | df_signals['Sell'])].iloc[-1:]
                    if not latest_signal.empty:
                        signal_time = latest_signal.index[0].tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M')
                        if latest_signal['Buy'].iloc[0]:
                            signal_placeholder.success(f"✅ BUY signal at {signal_time} IST")
                        elif latest_signal['Sell'].iloc[0]:
                            signal_placeholder.error(f"❌ SELL signal at {signal_time} IST")
                    else:
                        signal_placeholder.info("No recent signals")

                    data_placeholder.dataframe(df_signals.tail(10)[
                        ['open', 'high', 'low', 'close', 'EMA', 'RSI', 'MACD', 'Signal', 'Buy', 'Sell']
                    ])
            
            await asyncio.sleep(0.1)  # small delay to avoid tight loop
