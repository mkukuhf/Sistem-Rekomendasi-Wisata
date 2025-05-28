import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Rekomendasi Wisata Alam Sulawesi Utara",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .destination-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data and models"""
    try:
        # Load clean dataset
        df = pd.read_csv('./content/wisata_sulut_clean.csv')
        
        # Load models dari folder model
        with open('./content/model/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('./content/model/normalized_matrix.pkl', 'rb') as f:
            normalized_matrix = pickle.load(f)
        
        # Load destination features jika ada
        try:
            dest_features = pd.read_csv('./content/model/destination_features.csv')
        except:
            dest_features = None
            
        return df, vectorizer, normalized_matrix, dest_features
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Pastikan semua file model tersedia di /content/model/")
        return None, None, None, None

def create_user_vector(user_preferences, vectorizer):
    """Create user preference vector"""
    # Combine user preferences into a single string
    user_text = f"{user_preferences['kategori']} {user_preferences['jenis']} {user_preferences['aktivitas']} {user_preferences['fasilitas']}"
    
    # Transform using the same vectorizer
    user_vector = vectorizer.transform([user_text])
    
    # Save user preferences dan user vector (sesuai dengan file yang Anda buat)
    with open('./content/model/user_preferences.pkl', 'wb') as f:
        pickle.dump(user_preferences, f)
    
    with open('./content/model/user_vector.pkl', 'wb') as f:
        pickle.dump(user_vector, f)
    
    return user_vector

def get_recommendations(user_vector, normalized_matrix, df, top_k=5):
    """Get top K recommendations based on cosine similarity"""
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, normalized_matrix).flatten()
    
    # Get top K indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Create recommendations dataframe
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    
    # Save similarity results (sesuai dengan file yang Anda buat)
    similarity_results = pd.DataFrame({
        'destination_index': top_indices,
        'similarity_score': similarities[top_indices]
    })
    similarity_results.to_csv('./content/model/similarity_results.csv', index=False)
    
    # Save top recommendations (sesuai dengan file yang Anda buat)
    recommendations.to_csv('./content/model/top_recommendations.csv', index=False)
    recommendations.to_csv(f'./content/rekomendasi_wisata_top_{top_k}.csv', index=False)
    
    return recommendations

def display_destination_card(destination, similarity_score):
    """Display destination information in a card format"""
    with st.container():
        st.markdown(f"""
        <div class="destination-card">
            <h3 style="color: #333; margin-bottom: 1rem;">{destination['nama_destinasi']}</h3>
            <div class="similarity-score">
                Tingkat Kemiripan: {similarity_score:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**📍 Lokasi:** {destination['lokasi']}")
            st.write(f"**🏷️ Kategori:** {destination['kategori_utama']}")
            st.write(f"**🎯 Jenis Wisata:** {destination['jenis_wisata']}")
            if pd.notna(destination['deskripsi_singkat']):
                st.write(f"**📝 Deskripsi:** {destination['deskripsi_singkat']}")
            
        with col2:
            if pd.notna(destination['rating']):
                st.metric("⭐ Rating", f"{destination['rating']}/5")
            if pd.notna(destination['harga_tiket']):
                st.metric("💰 Harga Tiket", f"Rp {destination['harga_tiket']:,}")
        
        # Additional information in expandable section
        with st.expander("Informasi Detail"):
            col3, col4 = st.columns(2)
            with col3:
                if pd.notna(destination['aktivitas']):
                    st.write(f"**🏃 Aktivitas:** {destination['aktivitas']}")
                if pd.notna(destination['fasilitas']):
                    st.write(f"**🏢 Fasilitas:** {destination['fasilitas']}")
            with col4:
                if pd.notna(destination['aksesibilitas']):
                    st.write(f"**🚗 Aksesibilitas:** {destination['aksesibilitas']}")
                if pd.notna(destination['jam_operasional']):
                    st.write(f"**🕐 Jam Operasional:** {destination['jam_operasional']}")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏝️ Sistem Rekomendasi Wisata Alam</h1>
        <h3>Sulawesi Utara</h3>
        <p>Temukan destinasi wisata alam terbaik sesuai preferensi Anda</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, vectorizer, normalized_matrix, dest_features = load_data()
    
    if df is None:
        st.error("Gagal memuat data. Pastikan semua file tersedia:")
        st.error("- /content/wisata_sulut_clean.csv")
        st.error("- /content/model/tfidf_vectorizer.pkl") 
        st.error("- /content/model/normalized_matrix.pkl")
        return
    
    # Sidebar for user input
    st.sidebar.header("🎯 Preferensi Wisata Anda")
    st.sidebar.write("Masukkan preferensi Anda untuk mendapatkan rekomendasi terbaik:")
    
    # Get unique values for dropdowns
    categories = df['kategori_utama'].dropna().unique().tolist()
    jenis_wisata = df['jenis_wisata'].dropna().unique().tolist()
    
    # User input form
    with st.sidebar.form("preference_form"):
        st.subheader("Pilih Preferensi:")
        
        kategori = st.selectbox(
            "Kategori Wisata:",
            options=categories,
            help="Pilih kategori wisata yang Anda minati"
        )
        
        jenis = st.selectbox(
            "Jenis Wisata:",
            options=jenis_wisata,
            help="Pilih jenis wisata yang Anda sukai"
        )
        
        aktivitas = st.text_input(
            "Aktivitas yang Diinginkan:",
            placeholder="Contoh: hiking, snorkeling, photography",
            help="Masukkan aktivitas yang ingin Anda lakukan"
        )
        
        fasilitas = st.text_input(
            "Fasilitas yang Dibutuhkan:",
            placeholder="Contoh: toilet, parkir, penginapan",
            help="Masukkan fasilitas yang Anda perlukan"
        )
        
        top_k = st.slider("Jumlah Rekomendasi:", 1, 10, 5)
        
        submit_button = st.form_submit_button("🔍 Cari Rekomendasi")
    
    # Main content area
    if submit_button:
        # Create user preferences dictionary
        user_preferences = {
            'kategori': kategori,
            'jenis': jenis,
            'aktivitas': aktivitas,
            'fasilitas': fasilitas
        }
        
        # Show user preferences
        st.subheader("📋 Preferensi Anda:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Kategori:** {kategori}")
        with col2:
            st.info(f"**Jenis:** {jenis}")
        with col3:
            st.info(f"**Aktivitas:** {aktivitas}")
        with col4:
            st.info(f"**Fasilitas:** {fasilitas}")
        
        # Get recommendations
        with st.spinner("Sedang mencari rekomendasi terbaik untuk Anda..."):
            user_vector = create_user_vector(user_preferences, vectorizer)
            recommendations = get_recommendations(user_vector, normalized_matrix, df, top_k)
        
        # Display recommendations
        st.subheader(f"🎯 Top {top_k} Rekomendasi Wisata untuk Anda:")
        
        for idx, (_, destination) in enumerate(recommendations.iterrows()):
            st.markdown(f"### #{idx + 1}")
            display_destination_card(destination, destination['similarity_score'])
            st.markdown("---")
        
        # Show statistics
        st.subheader("📊 Statistik Rekomendasi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_similarity = recommendations['similarity_score'].mean()
            st.metric("Rata-rata Kemiripan", f"{avg_similarity:.2%}")
        
        with col2:
            max_similarity = recommendations['similarity_score'].max()
            st.metric("Kemiripan Tertinggi", f"{max_similarity:.2%}")
        
        with col3:
            min_similarity = recommendations['similarity_score'].min()
            st.metric("Kemiripan Terendah", f"{min_similarity:.2%}")
        
        # Visualization
        st.subheader("📈 Visualisasi Skor Kemiripan")
        
        fig = px.bar(
            x=range(1, len(recommendations) + 1),
            y=recommendations['similarity_score'],
            labels={'x': 'Ranking', 'y': 'Similarity Score'},
            title="Skor Kemiripan per Ranking"
        )
        fig.update_layout(
            xaxis_title="Ranking Rekomendasi",
            yaxis_title="Skor Kemiripan",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download recommendations
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Rekomendasi (CSV)",
            data=csv,
            file_name=f"rekomendasi_wisata_top_{top_k}.csv",
            mime="text/csv"
        )
    
    else:
        # Show dataset overview when no search is performed
        st.subheader("🗺️ Dataset Wisata Alam Sulawesi Utara")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Destinasi", len(df))
        with col2:
            st.metric("Total Kategori", df['kategori_utama'].nunique())
        with col3:
            avg_rating = df['rating'].mean()
            st.metric("Rata-rata Rating", f"{avg_rating:.1f}/5")
        
        # Show sample data
        st.subheader("📝 Contoh Data Destinasi:")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show model files status
        st.subheader("🔧 Status Model Files:")
        model_files = [
            './content/wisata_sulut_clean.csv',
            './content/model/tfidf_vectorizer.pkl',
            './content/model/scaler.pkl',
            './content/model/normalized_matrix.pkl',
            './content/model/destination_features.csv'
        ]
        
        for file_path in model_files:
            try:
                import os
                if os.path.exists(file_path):
                    st.success(f"✅ {file_path}")
                else:
                    st.warning(f"⚠️ {file_path} - File tidak ditemukan")
            except:
                st.warning(f"⚠️ {file_path} - Tidak dapat memeriksa file")
        
        # Distribution charts
        st.subheader("📊 Distribusi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = df['kategori_utama'].value_counts()
            fig1 = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribusi Kategori Wisata"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig2 = px.histogram(
                df,
                x='rating',
                title="Distribusi Rating Destinasi",
                nbins=20
            )
            fig2.update_layout(
                xaxis_title="Rating",
                yaxis_title="Jumlah Destinasi"
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()