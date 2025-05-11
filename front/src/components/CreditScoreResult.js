import React, { useState, useEffect } from 'react';
import { Container, Card, Row, Col, Alert, Spinner } from 'react-bootstrap';
import ShapValues from './ShapValues';
import ScoreCircle from './ScoreCircle';
import './CreditScoreResult.css';

const CreditScoreResult = ({ clientId, predictionData }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  // Fonction pour charger les résultats depuis l'API
  const fetchResult = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:5000/predict/${id}`);
      
      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Erreur lors de la récupération des données:', err);
      setError('Une erreur est survenue lors de la communication avec le serveur. Veuillez réessayer.');
    } finally {
      setLoading(false);
    }
  };

  // Utiliser predictionData si fourni, sinon charger depuis l'API
  useEffect(() => {
    if (predictionData) {
      setResult(predictionData);
    } else if (clientId && clientId.trim() !== '') {
      fetchResult(clientId);
    }
  }, [clientId, predictionData]);

  // Afficher le chargement
  if (loading) {
    return (
      <div className="d-flex justify-content-center my-5">
        <Spinner animation="border" role="status" variant="primary">
          <span className="visually-hidden">Chargement...</span>
        </Spinner>
        <span className="ms-2">Analyse en cours, veuillez patienter...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Une erreur est survenue</Alert.Heading>
        <p>{error}</p>
      </Alert>
    );
  }

  if (!result) {
    return null;
  }

  // Préparer les données et le style selon la prédiction
  const isAccepted = result.prediction === 0;
  const acceptProba = Math.round(result.proba[0] * 100);
  const rejectProba = Math.round(result.proba[1] * 100);
  const cardVariant = isAccepted ? 'success' : 'danger';
  const score = isAccepted ? acceptProba : (100 - rejectProba);
  const clientLabel = result.sk_id ? `Client #${result.sk_id}` : 'Nouvelle demande';
  
  return (
    <Container className="mt-4">
      <Card border={cardVariant} className="mb-4 result-card">
        <Card.Header className={`bg-${cardVariant} text-white`}>
          <h4 className="mb-0">
            {isAccepted ? 'Crédit Accepté' : 'Crédit Refusé'} - {clientLabel}
          </h4>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6} className="mb-3 mb-md-0">
              <Card className="h-100">
                <Card.Body className="text-center">
                  <h5 className="card-title mb-3">Score de Risque</h5>
                  <div className="d-flex justify-content-center">
                    <ScoreCircle score={score} />
                  </div>
                </Card.Body>
              </Card>
            </Col>
            <Col md={6}>
              <Card className="h-100">
                <Card.Body>
                  <h5 className="card-title mb-3">Probabilités</h5>
                  <div className="probability-container">
                    <div className="prob-label">Acceptation</div>
                    <div className="progress mb-2">
                      <div 
                        className="progress-bar bg-success" 
                        role="progressbar" 
                        style={{ width: `${acceptProba}%` }}
                        aria-valuenow={acceptProba} 
                        aria-valuemin="0" 
                        aria-valuemax="100"
                      >
                        {acceptProba}%
                      </div>
                    </div>
                    <div className="prob-label">Refus</div>
                    <div className="progress">
                      <div 
                        className="progress-bar bg-danger" 
                        role="progressbar" 
                        style={{ width: `${rejectProba}%` }} 
                        aria-valuenow={rejectProba} 
                        aria-valuemin="0" 
                        aria-valuemax="100"
                      >
                        {rejectProba}%
                      </div>
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          <ShapValues shapData={result.shap_values} />
        </Card.Body>
      </Card>
    </Container>
  );
};

export default CreditScoreResult; 