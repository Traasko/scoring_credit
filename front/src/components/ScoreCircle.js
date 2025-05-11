import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import './ScoreCircle.css';

// Enregistrer les composants Chart.js nécessaires
ChartJS.register(ArcElement, Tooltip, Legend);

const ScoreCircle = ({ score = 0 }) => {
  // Assurer que le score est entre 0 et 100
  const normalizedScore = Math.min(Math.max(parseInt(score) || 0, 0), 100);
  
  // Définir les couleurs en fonction du score
  let color;
  let label;
  
  if (normalizedScore >= 80) {
    color = '#198754';
    label = 'Excellent';
  } else if (normalizedScore >= 65) {
    color = '#79b62f';
    label = 'Bon';
  } else if (normalizedScore >= 50) {
    color = '#ffc107';
    label = 'Moyen';
  } else if (normalizedScore >= 30) {
    color = '#fd7e14';
    label = 'Faible';
  } else {
    color = '#dc3545';
    label = 'Risqué';
  }

  // Configurer les données pour le graphique en donut
  const data = {
    datasets: [
      {
        data: [normalizedScore, 100 - normalizedScore],
        backgroundColor: [color, '#e9ecef'],
        borderWidth: 0,
        cutout: '80%'
      }
    ]
  };

  // Options pour le graphique
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        enabled: false
      }
    },
    events: []
  };

  return (
    <div className="score-circle-container">
      <div className="score-circle">
        <Doughnut data={data} options={options} />
        <div className="score-overlay">
          <div className="score-value">{normalizedScore}</div>
          <div className="score-label">{label}</div>
        </div>
      </div>
    </div>
  );
};

export default ScoreCircle;
