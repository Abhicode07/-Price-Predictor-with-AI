"use client"
import { useEffect, useState } from "react";
import apiClient from "./apiClient";
import Chart from "./Chart";

interface CryptoData {
  timestamp: number;
  close: number;
}

const Home = () => {
  const [data, setData] = useState<CryptoData[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await apiClient.get("/crypto/historical", {
        params: { symbol: "BTC/USDT", timeframe: "1h" },
      });
      setData(response.data.data);
    };
    fetchData();
  }, []);

  return (
    <div>
      <h1>Crypto Price Prediction</h1>
      {data.length > 0 && <Chart data={data} />}
    </div>
  );
};

export default Home;
