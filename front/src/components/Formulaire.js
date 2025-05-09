import React, { useState } from "react";
import "../css/Formulaire.css";

const Formulaire = () => {
  const [formData, setFormData] = useState({
    AMT_INCOME_TOTAL: "",
    AMT_CREDIT: "",
    AMT_ANNUITY: "",
    AMT_GOODS_PRICE: "",
    DAYS_BIRTH: "",
    DAYS_EMPLOYED: "",
    NAME_EDUCATION_TYPE: "",
    NAME_FAMILY_STATUS: "",
    NAME_HOUSING_TYPE: "",
    OCCUPATION_TYPE: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);
  };

  return (
    <div className="form-wrapper">
      <h2>Formulaire Crédit</h2>
      <form onSubmit={handleSubmit} className="form-grid">
        {/* Ligne 1 */}
        <div className="form-group">
          <label>Revenu total du client<br /></label>
          <input name="AMT_INCOME_TOTAL" type="number" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Montant du crédit demandé<br /></label>
          <input name="AMT_CREDIT" type="number" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Annuité de remboursement<br /></label>
          <input name="AMT_ANNUITY" type="number" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Prix des biens pour le prêt<br /></label>
          <input name="AMT_GOODS_PRICE" type="number" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Âge du client<br/></label>
          <input name="DAYS_BIRTH" type="number" onChange={handleChange} />
        </div>

        {/* Ligne 2 */}
        <div className="form-group">
          <label>Jours d'emploi avant demande<br /></label>
          <input name="DAYS_EMPLOYED" type="number" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Niveau d'éducation<br /></label>
          <input name="NAME_EDUCATION_TYPE" type="text" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Statut familial<br /></label>
          <input name="NAME_FAMILY_STATUS" type="text" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Type de logement<br /></label>
          <input name="NAME_HOUSING_TYPE" type="text" onChange={handleChange} />
        </div>
        <div className="form-group">
          <label>Type d'emploi<br /></label>
          <input name="OCCUPATION_TYPE" type="text" onChange={handleChange} />
        </div>

        <button type="submit" className="form-button">Envoyer</button>
      </form>
    </div>
  );
};

export default Formulaire;
