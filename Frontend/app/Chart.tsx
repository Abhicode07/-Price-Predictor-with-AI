import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

interface ChartProps {
  data: { timestamp: number; close: number }[];
}

const Chart: React.FC<ChartProps> = ({ data }) => {
  const formattedData = data.map((entry) => ({
    time: new Date(entry.timestamp).toLocaleString(),
    price: entry.close,
  }));

  return (
    <LineChart width={800} height={400} data={formattedData}>
      <XAxis dataKey="time" />
      <YAxis dataKey="price" />
      <Tooltip />
      <Line type="monotone" dataKey="price" stroke="#82ca9d" />
    </LineChart>
  );
};

export default Chart;
