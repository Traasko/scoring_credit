import React from 'react';
import { Container, Row, Col, Navbar, Nav, Tab, Tabs } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import FormulaireId from './components/FormulaireId';
import Formulaire from './components/Formulaire';

function App() {
  return (
    <div className="App">
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand href="#home">Système de Scoring de Crédit</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              <Nav.Link href="#about">À propos</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container className="mt-4">
        <Row className="mb-4">
          <Col>
            <h1 className="text-center">Évaluation des Demandes de Crédit</h1>
            <p className="text-center lead">
              Utilisez notre système pour évaluer les demandes de crédit et comprendre 
              les facteurs qui influencent la décision.
            </p>
          </Col>
        </Row>

        <Tabs defaultActiveKey="id" id="credit-scoring-tabs" className="mb-4">
          <Tab eventKey="id" title="Évaluation par ID Client">
            <FormulaireId />
          </Tab>
          <Tab eventKey="form" title="Formulaire Complet">
            <Formulaire />
          </Tab>
        </Tabs>
      </Container>

      <footer className="bg-light py-4 mt-5">
        <Container>
          <p className="text-center text-muted mb-0">
            Système de Scoring de Crédit © {new Date().getFullYear()}
          </p>
        </Container>
      </footer>
    </div>
  );
}

export default App;
