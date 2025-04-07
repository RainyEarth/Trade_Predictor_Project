import React, { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

const TradePredictApp = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [plotData, setPlotData] = useState<any[]>([]);

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post("http://localhost:8000/api/predict", formData);
    setResult(response.data);

    if (response.data.chart_data) {
      setPlotData(response.data.chart_data); // optional if backend returns time series
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🧠 NeuralAditya Trade Predictor</h1>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="mb-4" />
      <button onClick={handleSubmit} className="bg-blue-600 text-white px-4 py-2 rounded">Predict</button>

      {result && (
        <div className="mt-6">
          <p className="text-lg font-medium">📈 Accuracy: {result.accuracy}</p>
          <p className="mt-2">🧮 Confusion Matrix:</p>
          <pre className="bg-gray-100 p-2 rounded">{JSON.stringify(result.confusion_matrix, null, 2)}</pre>

          <h2 className="text-xl font-semibold mt-6 mb-2">📊 Prediction Chart</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={plotData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="Close" stroke="#8884d8" name="Actual Close" />
              <Line type="monotone" dataKey="ARIMA_Pred" stroke="#82ca9d" name="ARIMA Prediction" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <footer className="text-center mt-12 text-gray-500 text-sm">
        © 2025 NeuralAditya. All rights reserved.
      </footer>
    </div>
  );
};

export default TradePredictApp;
