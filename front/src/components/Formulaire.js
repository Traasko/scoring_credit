import React, { useState } from "react";
import { Alert, Spinner } from 'react-bootstrap';
import CreditScoreResult from './CreditScoreResult';
import "../css/Formulaire.css";

const Formulaire = () => {
  const [formData, setFormData] = useState({
    AMT_CREDIT: "",
    AMT_GOODS_PRICE: "",
    AMT_ANNUITY: "",
    DAYS_BIRTH: "",
    DAYS_EMPLOYED: "",
    EXT_SOURCE_1: "0.5",
    EXT_SOURCE_2: "0.5",
    EXT_SOURCE_3: "0.5",
    NAME_EDUCATION_TYPE: "Secondary / secondary special",
    OCCUPATION_TYPE: "Laborers"
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const dataToSend = {
        ...formData,
        AMT_CREDIT: parseFloat(formData.AMT_CREDIT),
        AMT_GOODS_PRICE: parseFloat(formData.AMT_GOODS_PRICE),
        AMT_ANNUITY: parseFloat(formData.AMT_ANNUITY),
        DAYS_BIRTH: -(parseFloat(formData.DAYS_BIRTH)*365),
        DAYS_EMPLOYED: -parseFloat(formData.DAYS_EMPLOYED),
        EXT_SOURCE_1: parseFloat(formData.EXT_SOURCE_1),
        EXT_SOURCE_2: parseFloat(formData.EXT_SOURCE_2),
        EXT_SOURCE_3: parseFloat(formData.EXT_SOURCE_3)
      };
      
      console.log("Envoi des données:", dataToSend);
      
      const response = await fetch('http://localhost:5000/predict_new', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      });
      
      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Erreur lors de l'envoi du formulaire:", err);
      setError(`Erreur lors de l'analyse: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const educationOptions = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree"
  ];
  
  const occupationOptions = [
    "Laborers",
    "Core staff",
    "Managers",
    "Sales staff",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Cooking staff",
    "Security staff",
    "Cleaning staff",
    "Private service staff",
    "Low-skill Laborers",
    "Waiters/barmen staff",
    "Secretaries",
    "HR staff",
    "Realty agents",
    "IT staff"
  ];

  if (loading) {
    return (
      <div className="d-flex justify-content-center my-5">
        <Spinner animation="border" role="status" variant="primary" />
        <span className="ms-2">Analyse en cours, veuillez patienter...</span>
      </div>
    );
  }

  if (result) {
    return (
      <div>
        <button 
          onClick={() => setResult(null)} 
          className="btn btn-secondary mb-4"
        >
          Retour au formulaire
        </button>
        <CreditScoreResult predictionData={result} />
      </div>
    );
  }

  return (
    <div className="form-wrapper">
      <h2>Simuler une Demande de Crédit</h2>
      {error && (
        <Alert variant="danger" className="mb-4">
          {error}
        </Alert>
      )}
      <form onSubmit={handleSubmit} className="form-grid">
        <div className="form-group">
          <label>Montant du crédit demandé<span className="text-danger">*</span></label>
          <input 
            name="AMT_CREDIT" 
            type="number" 
            value={formData.AMT_CREDIT}
            onChange={handleChange} 
            required
          />
        </div>
        <div className="form-group">
          <label>Prix des biens<span className="text-danger">*</span></label>
          <input 
            name="AMT_GOODS_PRICE" 
            type="number" 
            value={formData.AMT_GOODS_PRICE}
            onChange={handleChange} 
            required
          />
        </div>
        <div className="form-group">
          <label>Annuité de remboursement<span className="text-danger">*</span></label>
          <input 
            name="AMT_ANNUITY" 
            type="number" 
            value={formData.AMT_ANNUITY}
            onChange={handleChange} 
            required
          />
        </div>

        <div className="form-group">
          <label>Âge<span className="text-danger">*</span></label>
          <input 
            name="DAYS_BIRTH" 
            type="number" 
            value={formData.DAYS_BIRTH}
            onChange={handleChange}
            placeholder="27"
            required
          />
        </div>
        <div className="form-group">
          <label>Jours d'emploi<span className="text-danger">*</span></label>
          <input 
            name="DAYS_EMPLOYED" 
            type="number" 
            value={formData.DAYS_EMPLOYED}
            onChange={handleChange}
            placeholder="1000"
            required
          />
          <small>Ex: 1000 pour environ 3 ans</small>
        </div>

        <div className="form-group">
          <label>Score externe 1 (0-1)</label>
          <input 
            name="EXT_SOURCE_1" 
            type="number" 
            step="0.01"
            min="0"
            max="1"
            value={formData.EXT_SOURCE_1}
            onChange={handleChange}
          />
        </div>
        <div className="form-group">
          <label>Score externe 2 (0-1)</label>
          <input 
            name="EXT_SOURCE_2" 
            type="number" 
            step="0.01"
            min="0"
            max="1"
            value={formData.EXT_SOURCE_2}
            onChange={handleChange}
          />
        </div>
        <div className="form-group">
          <label>Score externe 3 (0-1)</label>
          <input 
            name="EXT_SOURCE_3" 
            type="number" 
            step="0.01"
            min="0"
            max="1"
            value={formData.EXT_SOURCE_3}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label>Niveau d'éducation</label>
          <select 
            name="NAME_EDUCATION_TYPE" 
            value={formData.NAME_EDUCATION_TYPE}
            onChange={handleChange}
          >
            {educationOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>Type d'emploi</label>
          <select 
            name="OCCUPATION_TYPE" 
            value={formData.OCCUPATION_TYPE}
            onChange={handleChange}
          >
            {occupationOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>

        <button type="submit" className="form-button">Analyser la demande</button>
      </form>
    </div>
  );
};

export default Formulaire;
