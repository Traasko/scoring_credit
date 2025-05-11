import React from 'react';
import { Table, Alert, Card } from 'react-bootstrap';
import './ShapValues.css';

const ShapValues = ({ shapData }) => {
  if (!shapData || !shapData.feature_importance || shapData.error) {
    return (
      <Alert variant="info">
        Informations d'analyse d'impact non disponibles.
      </Alert>
    );
  }

  // Récupérer uniquement les 5 facteurs les plus importants
  const topFactors = shapData.feature_importance.slice(0, 5);

  // Formater les valeurs selon leur type
  const formatValue = (value, feature) => {
    if (feature === 'DAYS_BIRTH') {
      // Convertir jours en âge
      return `${Math.abs(Math.round(value / 365))} ans`;
    } else if (feature === 'DAYS_EMPLOYED') {
      // Convertir jours en années d'emploi
      const years = Math.abs(Math.round(value / 365));
      return `${years} an${years > 1 ? 's' : ''}`;
    } else if (feature.startsWith('AMT_')) {
      // Formater les montants
      return new Intl.NumberFormat('fr-FR', { 
        style: 'currency', 
        currency: 'EUR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(value);
    }
    return value;
  };

  return (
    <Card className="mb-4 shap-card">
      <Card.Header>
        <h5 className="mb-0">Facteurs influençant la décision</h5>
      </Card.Header>
      <Card.Body>
        {/* Graphique SHAP si disponible */}
        {shapData.shap_plot && (
          <div className="text-center mb-4">
            <img 
              src={`data:image/png;base64,${shapData.shap_plot}`}
              alt="Graphique SHAP" 
              className="img-fluid rounded shap-plot"
            />
          </div>
        )}

        {/* Tableau des facteurs importants */}
        <Table striped bordered hover responsive size="sm">
          <thead>
            <tr>
              <th>Facteur</th>
              <th>Valeur</th>
              <th>Impact</th>
              <th>Recommandation</th>
            </tr>
          </thead>
          <tbody>
            {topFactors.map((factor, index) => (
              <tr 
                key={index} 
                className={factor.impact === 'positif' ? 'table-success' : 'table-danger'}
              >
                <td>{factor.feature_name}</td>
                <td>{formatValue(factor.value, factor.feature)}</td>
                <td className={factor.impact === 'positif' ? 'text-success' : 'text-danger'}>
                  <strong>{factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1)}</strong>
                </td>
                <td>{factor.recommendation}</td>
              </tr>
            ))}
          </tbody>
        </Table>

        {/* Box de recommandation générale */}
        <Alert variant="info" className="mt-3">
          <h6 className="alert-heading">Comment améliorer votre score ?</h6>
          <p>
            Les facteurs listés ci-dessus sont ceux qui ont le plus d'impact sur votre score. 
            Pour augmenter vos chances d'obtenir un crédit, suivez les recommandations indiquées.
          </p>
        </Alert>
      </Card.Body>
    </Card>
  );
};

export default ShapValues; 