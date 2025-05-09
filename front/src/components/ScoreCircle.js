import React from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const ScoreCircle = ({ score }) => {
  return (
    <div style={{ width: 150, height: 150, margin: '2rem auto' }}>
      <CircularProgressbar
        value={score}
        text={`${score}%`}
        styles={buildStyles({
          pathColor: score >= 70 ? '#38a169' : score >= 40 ? '#ecc94b' : '#e53e3e',
          textColor: '#2d3748',
          trailColor: '#edf2f7',
          textSize: '18px',
          pathTransitionDuration: 1.2,
        })}
      />
    </div>
  );
};

export default ScoreCircle;
