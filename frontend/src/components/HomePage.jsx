import { Link } from 'react-router-dom'

export default function HomePage({ document }) {
  return (
    <div className="home-page">
      <div className="home-container">
        <header className="home-header">
          <h1>{document.title}</h1>
          <p className="version">Version: {document.version}</p>
        </header>

        <section className="home-content">
          <div className="info-section">
            <h2>About This Document</h2>
            <p>
              This is the {document.version} of the Kenya National Clinical Guidelines 
              for the Management of Diabetes Mellitus. These guidelines provide a 
              standardized approach to managing diabetes in Kenya, developed by the 
              National Diabetes Prevention and Control Program, Division of Non-communicable 
              Diseases, Ministry of Health, Kenya.
            </p>
          </div>

          <div className="info-section">
            <h2>How to Use This Guide</h2>
            <ol className="instructions-list">
              <li>
                Use the sidebar to navigate through chapters and sections.
              </li>
              <li>
                Click on any section to view its content with images and diagrams.
              </li>
              <li>
                On mobile devices, tap the menu icon to open the navigation sidebar.
              </li>
            </ol>
          </div>

          <div className="info-section">
            <h2>Source & Attribution</h2>
            <div className="source-info">
              <p><strong>Produced by:</strong> The National Diabetes Prevention and Control Program</p>
              <p><strong>Division:</strong> Division of Non-communicable Diseases, Ministry of Health, Kenya</p>
              <p><strong>Funded by:</strong> Ministry of Health, Kenya Diabetes Management and Information Centre 
                and World Diabetes Foundation (WDF)</p>
              <p><strong>Publication Year:</strong> 2018</p>
            </div>
          </div>

          <div className="info-section disclaimer-section">
            <h2>Disclaimer</h2>
            <div className="disclaimer-content">
              <p>
                Any part of this document may be freely reviewed, quoted, reproduced or 
                translated in full or in part so long as the source is acknowledged. 
                It is not for sale or for use in commercial purposes.
              </p>
              <p>
                These guidelines are intended for healthcare professionals and should be used 
                in conjunction with clinical judgment and patient-specific considerations. 
                The information provided is based on evidence available at the time of 
                publication and may be subject to updates as new evidence emerges.
              </p>
              <p>
                <strong>Important:</strong> These guidelines are not a substitute for 
                professional medical advice, diagnosis, or treatment. Always seek the 
                advice of qualified health providers with any questions regarding medical 
                conditions.
              </p>
            </div>
          </div>

          <div className="action-section">
            <Link to="/guidelines" className="start-button">
              Start Browsing Guidelines →
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}

