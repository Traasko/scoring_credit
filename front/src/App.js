import React from 'react';
import Formulaire from './components/Formulaire';
import FormulaireId from './components/FormulaireId';
import ScoreCircle from './components/ScoreCircle';

function App() {
  return (
    <div className="App">
      <Formulaire /> 
      <FormulaireId /> 
      <ScoreCircle score={75}/>

    </div>
  );
}

export default App;
