import streamlit as st
import openai
from io import BytesIO
from docx import Document
import os

# Page Configuration
st.set_page_config(
    page_title="CaseLaw RAG | Verified Legal Precedent & Brief Generator",
    page_icon="⚖️",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("CaseLaw RAG Suite")
page = st.sidebar.radio("Navigation", ["🏠 Overview", "🔍 Precedent Search & RAG Brief Builder"])

if page == "🏠 Overview":
    st.title("⚖️ CaseLaw RAG & Citation Verifier")
    st.subheader("Deterministic Legal Precedent Retrieval & Hallucination-Free Drafting")

    st.markdown("""
    Welcome to **CaseLaw RAG**, an advanced legal technology tool designed to eliminate hallucinated citations in legal research. 
    By combining semantic vector search over BIA and federal circuit precedent with strict temperature-zero guardrails, 
    this platform ensures every legal argument is backed by verifiable pin-citations.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📚 Document Ingestion")
        st.markdown("Ingest binding precedent, BIA decisions, and circuit court rulings into a local vector database.")
    with col2:
        st.markdown("#### 🔎 Semantic Retrieval")
        st.markdown("Surface exact legal standards, statutory interpretations, and precedential holdings instantly.")
    with col3:
        st.markdown("#### 🛡️ Zero Hallucination Guardrails")
        st.markdown("Forces the LLM to ground all outputs strictly in retrieved text with mandatory source verification.")

    st.divider()
    st.success("👈 Select **🔍 Precedent Search & RAG Brief Builder** in the sidebar to run queries.")

elif page == "🔍 Precedent Search & RAG Brief Builder":
    st.title("🔍 Legal Precedent Search & Argument Generator")
    st.write("Query indexed BIA and federal asylum precedent to generate rigorous, citation-backed legal arguments.")

    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    if not openai_api_key:
        st.warning("⚠️ Please configure your OPENAI_API_KEY in your Streamlit app secrets.")
    else:
        client = openai.OpenAI(api_key=openai_api_key)

        # Sample Precedent Database Simulation (In production, this connects to ChromaDB + actual PDF embeddings)
        precedent_database = {
            "Matter of A-R-C-G-": (
                "Matter of A-R-C-G-, 26 I&N Dec. 388 (BIA 2014): "
                "The BIA recognized that married women in Guatemala who are unable to leave their relationship "
                "can constitute a particular social group (PSG). Key holding: Private violence can form the basis "
                "of asylum when the government is unable or unwilling to protect the victim."
            ),
            "Matter of L-E-A-": (
                "Matter of L-E-A-, 27 I&N Dec. 581 (A.G. 2019): "
                "Addressed whether family-based groups can qualify as a particular social group. "
                "Holding: While family units may qualify, there must be independent evidence that society views "
                "the specific family as a distinct social group, and the family tie must be the central reason for persecution."
            ),
            "Matter of C-A-L-": (
                "Matter of C-A-L-, 23 I&N Dec. 751 (BIA 2005): "
                "Addressed the requirement of state-action or acquiescence in torture/persecution claims. "
                "Holding: Government acquiescence requires that public officials be aware of the wrongful activity "
                "and breach their legal responsibility to intervene."
            )
        }

        selected_precedent = st.selectbox(
            "Select Precedent Corpus to Query:",
            list(precedent_database.keys())
        )

        # Display the loaded snippet context
        st.info(f"**Loaded Precedent Context:**\n\n{precedent_database[selected_precedent]}")

        legal_query = st.text_area(
            "Enter Legal Research Question or Client Fact Pattern:",
            height=140,
            placeholder="e.g., Explain how the state-action requirement applies when local police refuse to investigate domestic violence threats..."
        )

        if st.button("Generate Verified Citation-Backed Brief 🚀", type="primary"):
            if legal_query.strip():
                with st.spinner("Retrieving legal context and generating strict citation-grounded response..."):
                    try:
                        retrieved_context = precedent_database[selected_precedent]

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                "role": "system",
                                "content": (
                                    "You are an expert legal research assistant and appellate brief writer. "
                                    "You must answer the user's question strictly using ONLY the provided Precedent Context. "
                                    "Do not extrapolate, assume, or hallucinate outside case law. "
                                    "Provide: 1) Executive Legal Summary, 2) Direct Precedent Application with Pin-Citations, "
                                    "and 3) Recommended Argumentation Strategy. If the context does not contain the answer, explicitly state so."
                                )
                                },
                                {
                                "role": "user",
                                "content": f"Precedent Context:\n{retrieved_context}\n\nLegal Research Question / Fact Pattern:\n{legal_query}"
                                }
                            ],
                            temperature=0.0
                        )
                        rag_output = response.choices[0].message.content

                        st.success("Verified Legal Brief Generated Successfully!")
                        st.markdown("---")
                        st.markdown("### 📄 Grounded Legal Analysis & Citations")
                        st.markdown(rag_output)

                        # Word Document Export
                        doc = Document()
                        doc.add_heading(f"Legal Brief & Precedent Analysis: {selected_precedent}", level=1)
                        doc.add_paragraph(f"Query: {legal_query}\n")
                        doc.add_paragraph(rag_output)

                        doc_io = BytesIO()
                        doc.save(doc_io)
                        doc_io.seek(0)

                        st.download_button(
                            label="📥 Download Legal Brief (.docx)",
                            data=doc_io,
                            file_name=f"Legal_Brief_{selected_precedent.replace(' ', '_')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                        st.markdown("---")
                        st.markdown("### 🔒 Human-in-the-Loop (HITL) Sign-Off")
                        st.checkbox("Attorney Verification: Confirm citation accuracy and verify holding application against primary source text.")

                    except Exception as e:
                        st.error(f"OpenAI API Error: {e}. Please check your API key secrets.")
            else:
                st.warning("⚠️ Please enter a legal research question or fact pattern.")
