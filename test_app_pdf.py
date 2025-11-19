#!/usr/bin/env python3
"""
Test version of the app with PDF download functionality
"""
import streamlit as st
import os
import json
import requests
from urllib.parse import quote
import time

# Configuration
BASE_DIR = "lifemodo_data"
os.makedirs(BASE_DIR, exist_ok=True)

def search_and_download_pdfs(query, max_results=3):
    """Recherche et télécharge des PDFs libres de droits depuis des sources académiques"""
    try:
        pdf_dir = os.path.join(BASE_DIR, "downloaded_pdfs")
        os.makedirs(pdf_dir, exist_ok=True)

        downloaded_pdfs = []

        # Sources de PDFs libres de droits
        sources = [
            {
                "name": "arXiv",
                "search_url": f"http://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending",
                "pdf_base": "https://arxiv.org/pdf/"
            }
        ]

        for source in sources:
            try:
                st.info(f"🔍 Recherche sur {source['name']}...")

                response = requests.get(source["search_url"], timeout=10)
                response.raise_for_status()

                if source["name"] == "arXiv":
                    # Parser XML arXiv
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.content)

                    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")[:max_results]:
                        title_elem = entry.find(".//{http://www.w3.org/2005/Atom}title")
                        id_elem = entry.find(".//{http://www.w3.org/2005/Atom}id")

                        if title_elem is not None and id_elem is not None:
                            title = title_elem.text.strip()
                            arxiv_id = id_elem.text.split('/')[-1]
                            pdf_url = f"{source['pdf_base']}{arxiv_id}.pdf"

                            # Télécharger le PDF
                            pdf_response = requests.get(pdf_url, timeout=30)
                            if pdf_response.status_code == 200:
                                pdf_filename = f"arxiv_{arxiv_id}.pdf"
                                pdf_path = os.path.join(pdf_dir, pdf_filename)

                                with open(pdf_path, 'wb') as f:
                                    f.write(pdf_response.content)

                                downloaded_pdfs.append({
                                    "title": title,
                                    "source": source["name"],
                                    "path": pdf_path,
                                    "url": pdf_url
                                })

                                st.success(f"✅ Téléchargé: {title[:50]}...")
                                time.sleep(1)  # Respect rate limits

            except Exception as e:
                st.warning(f"Erreur avec {source['name']}: {str(e)}")
                continue

        return downloaded_pdfs

    except Exception as e:
        st.error(f"Erreur recherche PDFs: {str(e)}")
        return []

def main():
    st.title("🧬 LifeModo AI Lab - Test PDF Download")

    st.header("🤖 Agent IA - Mistral (Test Mode)")

    with st.expander("🧠 Guide de l'Agent Mistral"):
        st.markdown("""
        ## 🤖 Agent IA Multimodal - Mistral 7B

        ### 🎯 **Rôle de l'Agent**
        L'agent Mistral peut maintenant télécharger automatiquement des PDFs depuis arXiv et Papers with Code quand vous lui demandez.
        """)

    # Section utilisation de l'agent
    st.subheader("🧠 Test de l'Agent IA")

    user_input = st.text_area(
        "Posez votre question à l'agent IA :",
        placeholder="Ex: 'Télécharge-moi des PDFs sur le machine learning'",
        height=100
    )

    if st.button("🚀 Demander à l'agent", type="primary"):
        if user_input.strip():
            # Détecter les demandes de téléchargement de PDFs
            pdf_keywords = ["télécharge", "download", "pdf", "document", "paper", "article", "recherche", "cherche"]
            is_pdf_request = any(keyword in user_input.lower() for keyword in pdf_keywords)

            if is_pdf_request:
                st.info("📄 Demande de PDF détectée - Recherche et téléchargement automatique...")

                # Extraire la requête de recherche du message utilisateur
                search_query = user_input.lower()
                # Nettoyer la requête pour la recherche
                for keyword in pdf_keywords:
                    search_query = search_query.replace(keyword, "")
                search_query = search_query.strip()

                if not search_query:
                    search_query = "machine learning"  # Requête par défaut

                st.write(f"🔍 Recherche de PDFs sur : '{search_query}'")

                # Rechercher et télécharger les PDFs
                downloaded_pdfs = search_and_download_pdfs(search_query, max_results=2)

                if downloaded_pdfs:
                    st.success(f"✅ {len(downloaded_pdfs)} PDFs téléchargés avec succès!")

                    # Afficher les PDFs téléchargés
                    st.markdown("### 📚 PDFs Téléchargés:")
                    for pdf in downloaded_pdfs:
                        st.write(f"📄 **{pdf['title']}**")
                        st.write(f"Source: {pdf['source']}")
                        st.write(f"Chemin: `{pdf['path']}`")

                        # Bouton de téléchargement
                        with open(pdf['path'], 'rb') as f:
                            st.download_button(
                                label=f"💾 Télécharger {os.path.basename(pdf['path'])}",
                                data=f,
                                file_name=os.path.basename(pdf['path']),
                                mime="application/pdf"
                            )

                    # Réponse simulée de Mistral
                    st.markdown("### 🤖 Réponse de l'Agent Mistral:")
                    st.markdown(f"J'ai automatiquement téléchargé {len(downloaded_pdfs)} documents PDF sur '{search_query}' depuis arXiv. Ces documents peuvent être utilisés pour enrichir votre dataset multimodal ou pour entraîner de nouveaux modèles d'IA.")

                else:
                    st.warning("⚠️ Aucun PDF trouvé pour cette requête. Essaie avec des termes plus spécifiques.")

            else:
                # Réponse normale simulée
                st.markdown("### 🤖 Réponse de l'Agent Mistral:")
                st.markdown("Je suis l'agent IA Mistral. Pour télécharger des PDFs, dites-moi quelque chose comme 'Télécharge-moi des PDFs sur le machine learning'.")
        else:
            st.warning("Veuillez entrer une question.")

if __name__ == "__main__":
    main()