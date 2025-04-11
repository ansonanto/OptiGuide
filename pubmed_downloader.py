import os
import json
import time
import hashlib
import streamlit as st
from Bio import Entrez
import pdfplumber
from tqdm import tqdm
from metapub import PubMedFetcher
import requests
from bs4 import BeautifulSoup
import subprocess
import sys
import os
import re
import urllib.request
from urllib.parse import urljoin

# Configuration
RESULTS_FOLDER = "./results/"
PUBMED_ERRORS_FILEPATH = "./unfetch_pmids.txt"
DOWNLOADED_PUBMEDS_FILEPATH = "./downloaded_pubmeds.tsv"
EMAIL_ID = "user@example.com"  # Replace with your email

# PMC Base URLs
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

# Initialize PubMed fetcher
FETCH = PubMedFetcher()
Entrez.email = EMAIL_ID

def initial_setup():
    """
    Initial Filesystem setup if not already exists.
    """
    # Check if the results folder exists, if not, create it
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Creating folder: {RESULTS_FOLDER}")
        os.makedirs(RESULTS_FOLDER)

    # Create empty error file if it does not exist
    if not os.path.exists(PUBMED_ERRORS_FILEPATH):
        print(f"Creating errors file: {PUBMED_ERRORS_FILEPATH}")
        with open(PUBMED_ERRORS_FILEPATH, 'w') as f:
            f.write('')
            
    # Create downloaded PMIDs file if it does not exist
    if not os.path.exists(DOWNLOADED_PUBMEDS_FILEPATH):
        print(f"Creating downloaded PMIDs file: {DOWNLOADED_PUBMEDS_FILEPATH}")
        with open(DOWNLOADED_PUBMEDS_FILEPATH, 'w') as f:
            f.write('pmid\tpmc_id\ttitle\n')  # Header row

def append_to_error_file(pmid):
    """
    Update the status file for the PMID as the error in downloading the article.
    """
    if os.path.exists(PUBMED_ERRORS_FILEPATH):
        with open(PUBMED_ERRORS_FILEPATH, 'r') as f:
            existing_pmids = f.read().splitlines()
    else:
        existing_pmids = []

    if pmid not in existing_pmids:
        with open(PUBMED_ERRORS_FILEPATH, 'a') as f:
            f.write(f"{pmid}\n")
        print(f"PMID {pmid} added to error file.")
    else:
        print(f"PMID {pmid} is already in the error file. Skipping appending.")

def get_pubmed_search_query(search_term):
    """
    Generate a PubMed API search query for a given search term.
    """
    query = f"""
        (
            ({search_term}[Title/Abstract])
        ) 
        AND english[Language] 
        AND free full text[Filter]
        NOT (books and documents[pt] OR review[pt])
    """
    return query

def is_article_already_downloaded(pmid):
    """
    Check if the article is already downloaded.
    """
    expected_pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
    return os.path.exists(expected_pdf_path)

def get_pmc_id_from_pmid(pmid):
    """
    Get the PMC ID for a given PubMed ID using Entrez.
    """
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", linkname="pubmed_pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()
        
        if record[0]["LinkSetDb"] and record[0]["LinkSetDb"][0]["Link"]:
            pmc_id = record[0]["LinkSetDb"][0]["Link"][0]["Id"]
            return f"PMC{pmc_id}"
        return None
    except Exception as e:
        print(f"Error getting PMC ID for PMID {pmid}: {str(e)}")
        return None

def download_from_pmc_direct(pmid, pmc_id):
    """
    Download a paper directly from PubMed Central using the PMC ID.
    """
    try:
        # Try to get the PDF URL from the PMC page
        pmc_url = f"{PMC_BASE_URL}{pmc_id}/"
        response = requests.get(pmc_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for PDF link
        pdf_link = None
        for a in soup.find_all('a', href=True):
            if 'pdf' in a['href'].lower() and ('.pdf' in a['href'].lower() or '/pdf/' in a['href'].lower()):
                pdf_link = a['href']
                break
        
        if pdf_link:
            # Make sure it's a full URL
            if not pdf_link.startswith('http'):
                pdf_link = urljoin(pmc_url, pdf_link)
            
            # Download the PDF
            pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
            urllib.request.urlretrieve(pdf_link, pdf_path)
            
            # Verify the file was downloaded and is a valid PDF
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:  # Basic size check
                # Get article metadata for better filename
                article = FETCH.article_by_pmid(pmid)
                if article and article.title:
                    # Create a safe filename from the title
                    safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '' for c in article.title])
                    safe_title = safe_title[:100]  # Limit length
                    new_filename = f"{pmid}_{safe_title}.pdf"
                    os.rename(pdf_path, os.path.join(RESULTS_FOLDER, new_filename))
                    
                    # Record the download in the TSV file
                    with open(DOWNLOADED_PUBMEDS_FILEPATH, 'a') as f:
                        f.write(f"{pmid}\t{pmc_id}\t{article.title}\n")
                    
                    return True
        return False
    except Exception as e:
        print(f"Error downloading from PMC direct for PMID {pmid}: {str(e)}")
        return False

def download_using_mcp(pmid):
    """
    Try to download using the MCP tool.
    """
    try:
        # This is a placeholder for the MCP tool integration
        # In a real implementation, you would call the appropriate MCP API here
        return False
    except Exception as e:
        print(f"Error using MCP for PMID {pmid}: {str(e)}")
        return False

def download_pubmed_articles(pmids, progress_bar=None):
    """
    Download the specified PubMed articles using multiple methods.
    """
    downloaded_count = 0
    
    for i, pmid in enumerate(pmids):
        if progress_bar:
            progress_bar.progress((i + 1) / len(pmids), text=f"Downloading article {i+1}/{len(pmids)}")
        
        if is_article_already_downloaded(pmid):
            print(f"PMID {pmid} already exists. Skipping download.")
            downloaded_count += 1
            continue  # Skip download if already exists

        print(f"Downloading article for PMID: {pmid}")
        success = False
        
        # Method 1: Try to get PMC ID and download directly from PMC
        pmc_id = get_pmc_id_from_pmid(pmid)
        if pmc_id:
            print(f"Found PMC ID {pmc_id} for PMID {pmid}")
            success = download_from_pmc_direct(pmid, pmc_id)
        
        # Method 2: If PMC direct download failed, try pubmed2pdf
        if not success:
            try:
                temp_error_file = os.path.join(RESULTS_FOLDER, f"pubmed2pdf_errors_{pmid}.txt")
                result = subprocess.run(
                    [sys.executable, "-m", "pubmed2pdf", "pdf", 
                     f"--out={RESULTS_FOLDER}", 
                     f"--errors={temp_error_file}", 
                     f"--pmids={pmid}"],
                    capture_output=True,
                    text=True,
                    timeout=60  # Set a timeout to avoid hanging
                )
                
                # Check if the download was successful
                if is_article_already_downloaded(pmid):
                    success = True
                    # Rename the file to include the PMID for easier identification
                    article = FETCH.article_by_pmid(pmid)
                    if article and article.title:
                        # Create a safe filename from the title
                        safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '' for c in article.title])
                        safe_title = safe_title[:100]  # Limit length
                        new_filename = f"{pmid}_{safe_title}.pdf"
                        os.rename(
                            os.path.join(RESULTS_FOLDER, f"{pmid}.pdf"),
                            os.path.join(RESULTS_FOLDER, new_filename)
                        )
                        
                        # Record the download in the TSV file
                        with open(DOWNLOADED_PUBMEDS_FILEPATH, 'a') as f:
                            f.write(f"{pmid}\t{pmc_id or ''}\t{article.title}\n")
                
                # Check and process error file
                if os.path.exists(temp_error_file):
                    with open(temp_error_file, 'r') as f:
                        error_pmids = f.read().splitlines()
                    for error_pmid in error_pmids:
                        append_to_error_file(error_pmid)
                    # Remove the temp error file
                    os.remove(temp_error_file)
            except Exception as e:
                print(f"Error using pubmed2pdf for PMID {pmid}: {str(e)}")
        
        # Method 3: Try MCP tool as a last resort
        if not success:
            success = download_using_mcp(pmid)
        
        if success:
            downloaded_count += 1
            print(f"Successfully downloaded PMID {pmid}")
        else:
            print(f"Failed to download PMID: {pmid}")
            append_to_error_file(pmid)
    
    return downloaded_count

def download_articles_by_keywords(keywords, max_articles_per_keyword=5):
    """
    For the given keywords, identify and download PubMed articles.
    """
    initial_setup()
    
    start = time.time()
    total_downloaded = 0
    
    # Create download log for the UI
    download_log = []
    download_log.append("Download Log\n")
    
    for keyword in keywords:
        query = get_pubmed_search_query(keyword)
        try:
            pmids = FETCH.pmids_for_query(query, retmax=max_articles_per_keyword)
            pmids = list(set(pmids))  # Remove duplicates
            
            log_message = f"Found {len(pmids)} articles for keyword '{keyword}'."
            print(log_message)
            download_log.append(log_message)
            
            if pmids:
                downloaded = download_pubmed_articles(pmids)
                total_downloaded += downloaded
                log_message = f"Downloaded {downloaded} articles for keyword '{keyword}'"
                print(log_message)
                download_log.append(log_message)
            else:
                log_message = f"No articles found for keyword '{keyword}'"
                print(log_message)
                download_log.append(log_message)
                
        except Exception as e:
            log_message = f"Error searching for keyword '{keyword}': {str(e)}"
            print(log_message)
            download_log.append(log_message)
    
    end = time.time()
    log_message = f"Time to Process: {round(end-start, 2)} secs"
    print(log_message)
    download_log.append(log_message)
    
    return total_downloaded, "\n".join(download_log)

# Streamlit interface for the PubMed downloader
def pubmed_downloader_ui():
    st.header("PubMed Article Downloader")
    
    st.write("""
    This tool allows you to download research papers from PubMed based on keywords.
    The downloaded papers will be saved to the 'results' folder and can be queried using the RAG system.
    """)
    
    # Input for keywords
    st.subheader("Enter Keywords")
    keywords_input = st.text_area(
        "Enter keywords (one per line):",
        height=150,
        help="Enter each keyword or phrase on a new line. For example:\nExercise and Hypertension\nExercise and Diabetes"
    )
    
    # Number of articles per keyword
    max_articles = st.slider(
        "Maximum articles per keyword:",
        min_value=1,
        max_value=20,
        value=5,
        help="Limit the number of articles to download for each keyword"
    )
    
    # Create a placeholder for the download log
    download_log_placeholder = st.empty()
    
    # Download button
    if st.button("Download Articles"):
        if not keywords_input.strip():
            st.error("Please enter at least one keyword.")
            return
        
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
        with st.spinner(f"Downloading articles for {len(keywords)} keywords..."):
            progress_bar = st.progress(0)
            downloaded, log_text = download_articles_by_keywords(keywords, max_articles)
            progress_bar.progress(1.0)
            
            # Display the download log
            download_log_placeholder.text_area("Download Log", log_text, height=400)
        
        if downloaded > 0:
            st.success(f"Successfully downloaded {downloaded} articles!")
            st.info("The articles are now available in the 'results' folder and can be processed by the RAG system.")
        else:
            st.warning("No articles were downloaded. This could be due to no matching articles or download failures.")
            st.info("Check the download log for more details on any errors.")

if __name__ == "__main__":
    download_articles_by_keywords(["Invasive Breast Cancer", "Inflammatory Breast Cancer"], 5)
