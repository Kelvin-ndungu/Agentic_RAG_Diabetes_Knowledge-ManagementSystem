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

          <div className="info-section">
            <h2>Developer Information</h2>
            <div className="developer-info">
              <p><strong>Developed by:</strong> Kelvin Ndungu Kinyanjui</p>
              <p><strong>Mobile:</strong> <a href="tel:+254713281876">+254 713 281 876</a></p>
              <p><strong>Email:</strong> <a href="mailto:Kinyanjuikelvin047@gmail.com">Kinyanjuikelvin047@gmail.com</a></p>
            </div>
          </div>

          <div className="info-section">
            <h2>Educational Purpose</h2>
            <div className="educational-purpose">
              <p>
                This application has been developed for <strong>educational purposes</strong> to demonstrate 
                the potential of Artificial Intelligence (AI) in creating natural language chat interfaces 
                for knowledge management systems.
              </p>
              <p>
                The system showcases how AI can be integrated with knowledge bases to provide:
              </p>
              <ul>
                <li>Question-and-answer chatbots that understand natural language queries</li>
                <li>Semantic search capabilities that retrieve relevant information based on meaning, not just keywords</li>
                <li>Integration of structured knowledge bases with conversational AI interfaces</li>
                <li>Retrieval-Augmented Generation (RAG) systems for accurate, source-cited responses</li>
              </ul>
              <p>
                This project serves as a demonstration of how modern AI technologies can make complex 
                medical knowledge more accessible through intuitive interfaces.
              </p>
            </div>
          </div>

          <div className="info-section disclaimer-section">
            <h2>Disclaimer & Terms</h2>
            <div className="disclaimer-content">
              <p>
                <strong>Source Material:</strong> The content presented in this application is derived from 
                the <strong>Kenya National Clinical Guidelines for the Management of Diabetes Mellitus, 
                2nd Edition (2018)</strong>, produced by the National Diabetes Prevention and Control Program, 
                Division of Non-communicable Diseases, Ministry of Health, Kenya.
              </p>
              <p>
                Any part of the original document may be freely reviewed, quoted, reproduced or 
                translated in full or in part so long as the source is acknowledged. 
                It is not for sale or for use in commercial purposes.
              </p>
              <p>
                <strong>Medical Disclaimer:</strong> These guidelines are intended for healthcare professionals 
                and should be used in conjunction with clinical judgment and patient-specific considerations. 
                The information provided is based on evidence available at the time of 
                publication and may be subject to updates as new evidence emerges.
              </p>
              <p>
                <strong>Important:</strong> This application and the information it provides are <strong>not a substitute 
                for professional medical advice, diagnosis, or treatment</strong>. Always seek the 
                advice of qualified health providers with any questions regarding medical 
                conditions. Never disregard professional medical advice or delay in seeking it 
                because of something you have read or accessed through this application.
              </p>
              <p>
                <strong>AI Limitations:</strong> While this system uses advanced AI technologies, the responses 
                are generated based on the source material and may not reflect the most current 
                medical knowledge. Users should verify critical information with authoritative 
                sources and consult healthcare professionals for medical decisions.
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

