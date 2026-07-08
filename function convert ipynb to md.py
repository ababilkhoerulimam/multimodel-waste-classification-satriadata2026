import json

def ipynb_to_markdown_code_only(ipynb_path, output_md_path):
    """
    Membaca file .ipynb dan mengekstrak semua sel kode ke dalam file .md
    dengan format code block python.
    """
    try:
        # Membaca file Jupyter Notebook
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        markdown_lines = []
        
        # Iterasi setiap sel di dalam notebook
        for cell in notebook.get('cells', []):
            # Hanya ambil sel yang bertipe 'code'
            if cell.get('cell_type') == 'code':
                # Menggabungkan baris kode di dalam sel
                code_content = "".join(cell.get('source', []))
                
                # Lewati jika sel kode kosong
                if not code_content.strip():
                    continue
                
            
                # Bungkus kode ke dalam format code block Markdown
                markdown_lines.append("```python")
                markdown_lines.append(code_content)
                markdown_lines.append("```\n")
        
        # Menulis hasil ke file Markdown
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
            
        print(f"Berhasil! Kode telah diekstrak ke: {output_md_path}")
        
    except FileNotFoundError:
        print(f"Error: File '{ipynb_path}' tidak ditemukan.")
    except json.JSONDecodeError:
        print("Error: File bukan format JSON/ipynb yang valid.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# --- Cara Penggunaan ---
ipynb_to_markdown_code_only("satria-data-kaggle.ipynb", "satria-data-kaggle.md")
