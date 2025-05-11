import React, { useState } from 'react';
import { Form, Button, Card, Container, Row, Col } from 'react-bootstrap';
import CreditScoreResult from './CreditScoreResult';

const FormulaireId = () => {
  const [clientId, setClientId] = useState('');
  const [submittedId, setSubmittedId] = useState('');
  const [validated, setValidated] = useState(false);

  const handleSubmit = (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    
    setValidated(true);
    
    if (form.checkValidity() === false) {
      event.stopPropagation();
      return;
    }
    
    // Enregistrer l'ID soumis pour afficher le résultat
    setSubmittedId(clientId);
  };

  return (
    <Container className="my-4">
      <Row className="justify-content-center">
        <Col md={8} lg={6}>
          <Card>
            <Card.Header className="bg-primary text-white">
              <h2 className="h5 mb-0">Évaluation de crédit par ID client</h2>
            </Card.Header>
            <Card.Body>
              <Form noValidate validated={validated} onSubmit={handleSubmit}>
                <Form.Group className="mb-3" controlId="formClientId">
                  <Form.Label>Identifiant du client (SK_ID_CURR)</Form.Label>
                  <Form.Control 
                    type="number" 
                    placeholder="Ex: 100001" 
                    value={clientId}
                    onChange={(e) => setClientId(e.target.value)}
                    required
                    min="100000"
                  />
                  <Form.Control.Feedback type="invalid">
                    Veuillez saisir un ID client valide (minimum 100000).
                  </Form.Control.Feedback>
                  <Form.Text className="text-muted">
                    Entrez l'identifiant du client pour évaluer sa demande de crédit.
                  </Form.Text>
                </Form.Group>
                
                <Button variant="primary" type="submit">
                  Évaluer la demande
                </Button>
              </Form>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {submittedId && <CreditScoreResult clientId={submittedId} />}
    </Container>
  );
};

export default FormulaireId;