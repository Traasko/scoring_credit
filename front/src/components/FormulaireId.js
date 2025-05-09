import React, { useState } from "react";
import "../css/Formulaire.css";

const FormulaireId = () => {
  const [formData, setFormData] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);
  };

  return (
    <div className="form-wrapper">
      <h2>ID pour la demande de prêt</h2>
      <form onSubmit={handleSubmit} className="form-grid">
        <div className="form-group">
          <label>ID client<br/></label>
          <input name="ID_CURRENT" onChange={handleChange} />
        </div>
        <button type="submit" className="form-button">Envoyer</button>
      </form>
    </div>
  );
};

export default FormulaireId;