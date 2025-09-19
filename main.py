from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.exceptions import HTTPException  # Add this import
import pandas as pd
from reportlab.pdfgen import canvas
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import re
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import http.client
import json
from dotenv import load_dotenv


warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key


# Load BigBasket data with proper column mapping
def load_products():
    try:
        df = pd.read_csv('data/BigBasket.csv')
        print("Data loaded successfully!")
        print(f"Number of products: {len(df)}")

        # Map BigBasket columns to our expected names
        df['price'] = df['DiscountPrice'].fillna(df['Price'])
        df['weight'] = df['Quantity']
        df['name'] = df['ProductName']
        df['shop'] = df['Brand']
        df['category'] = df['Category']
        df['image_url'] = df['Image_Url']

        # Add popularity score (simulated)
        np.random.seed(42)
        df['popularity'] = np.random.randint(1, 100, len(df))

        # Create ID if not exists
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)

        # Select only the columns we need
        df = df[['id', 'name', 'price', 'shop', 'weight', 'category', 'image_url', 'popularity']].copy()

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a sample DataFrame if file doesn't exist
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Apple', 'Bread', 'Milk', 'Eggs', 'Chicken'],
            'price': [1.50, 2.75, 3.25, 4.50, 8.99],
            'shop': ['SuperMart', 'Bakery Direct', 'SuperMart', 'Fresh Foods', 'Meat Market'],
            'weight': ['0.2kg', '0.5kg', '1L', '12ct', '1kg'],
            'category': ['Produce', 'Bakery', 'Dairy', 'Dairy', 'Meat'],
            'image_url': ['apple.jpg', 'bread.jpg', 'milk.jpg', 'eggs.jpg', 'chicken.jpg'],
            'popularity': [85, 72, 90, 88, 79]
        })


# Load property data
# Load property data
# Load property data
def load_properties():
    try:
        # Check if file exists first
        file_path = 'data/world_real_estate_data(147k).csv'
        if not os.path.exists(file_path):
            print(f"Property data file not found at: {file_path}")
            print("Current working directory:", os.getcwd())
            print("Directory contents:", os.listdir('.'))
            if os.path.exists('data'):
                print("Data directory contents:", os.listdir('data'))
            return get_sample_property_data()

        df = pd.read_csv(file_path)
        print("Property data loaded successfully!")
        print(f"Number of properties: {len(df)}")
        print("Available countries:", df['country'].unique()[:10])  # Show first 10 countries
        print("Available columns:", df.columns.tolist())  # Debug: show all columns

        # Check for price column with different possible names
        price_column = None
        possible_price_columns = ['price_in_USD', 'price_in_usd', 'price', 'price_usd', 'price_dollar', 'usd_price']

        for col in possible_price_columns:
            if col in df.columns:
                price_column = col
                print(f"Found price column: {price_column}")
                break

        if not price_column:
            print("Warning: No price column found in property data")
            return get_sample_property_data()

        # Rename the price column to 'price_in_usd' for consistency
        if price_column != 'price_in_USD':
            df = df.rename(columns={price_column: 'price_in_USD'})
            print(f"Renamed column '{price_column}' to 'price_in_USD'")

        # Check if required columns exist
        required_columns = ['title', 'country', 'price_in_USD']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in property data: {missing_columns}")
            return get_sample_property_data()

        # Data cleaning and preprocessing
        df['building_construction_year'] = df['building_construction_year'].fillna(2000)
        df['building_total_floors'] = df['building_total_floors'].fillna(1)
        df['apartment_floor'] = df['apartment_floor'].fillna(1)
        df['apartment_rooms'] = df['apartment_rooms'].fillna(1)
        df['apartment_bedrooms'] = df['apartment_bedrooms'].fillna(1)
        df['apartment_bathrooms'] = df['apartment_bathrooms'].fillna(1)

        # Handle area columns with units (like '120 m²')
        if 'apartment_total_area' in df.columns:
            print("Processing apartment_total_area column...")
            df['apartment_total_area_numeric'] = df['apartment_total_area'].apply(extract_numeric_area)
            df['apartment_total_area'] = df['apartment_total_area_numeric'].fillna(
                df['apartment_total_area_numeric'].median())

        if 'apartment_living_area' in df.columns:
            print("Processing apartment_living_area column...")
            df['apartment_living_area_numeric'] = df['apartment_living_area'].apply(extract_numeric_area)
            df['apartment_living_area'] = df['apartment_living_area_numeric'].fillna(
                df['apartment_living_area_numeric'].median())

        # Create a quality score based on various features
        df['quality_score'] = calculate_property_quality_score(df)

        return df

    except Exception as e:
        print(f"Error loading property data: {e}")
        import traceback
        traceback.print_exc()
        return get_sample_property_data()

def get_sample_property_data():
    """Return sample property data with multiple countries"""
    return pd.DataFrame({
        'title': ['Luxury Apartment in Istanbul', 'Beach House in Antalya', 'City Center Apartment',
                  'Villa in Dubai', 'Apartment in New York', 'House in London', 'Condominium in Tokyo'],
        'country': ['Turkey', 'Turkey', 'Turkey', 'UAE', 'USA', 'UK', 'Japan'],
        'location': ['Istanbul', 'Antalya', 'Ankara', 'Dubai', 'New York', 'London', 'Tokyo'],
        'price_in_USD': [250000, 180000, 120000, 500000, 800000, 600000, 400000],
        'apartment_total_area': [120, 95, 75, 200, 85, 150, 70],
        'quality_score': [8.5, 7.2, 6.8, 9.0, 8.8, 8.2, 9.2],
        'building_construction_year': [2015, 2010, 2005, 2020, 2018, 2012, 2019],
        'apartment_rooms': [3, 2, 2, 4, 3, 4, 2],
        'apartment_bedrooms': [2, 2, 1, 3, 2, 3, 2],
        'apartment_bathrooms': [2, 1, 1, 3, 2, 2, 2],
        'url': ['#', '#', '#', '#', '#', '#', '#'],
        'image': ['#', '#', '#', '#', '#', '#', '#']
    })


def calculate_property_quality_score(df):
    try:
        # Normalize features
        scaler = StandardScaler()

        # Features that contribute to quality with proper NaN handling
        construction_year = df['building_construction_year'].fillna(2000).values.reshape(-1, 1)
        construction_year_norm = scaler.fit_transform(construction_year).flatten()

        # Use the numeric area values with proper NaN handling
        if 'apartment_total_area_numeric' in df.columns:
            total_area = df['apartment_total_area_numeric'].fillna(
                df['apartment_total_area_numeric'].median()).values.reshape(-1, 1)
        else:
            total_area = df['apartment_total_area'].fillna(df['apartment_total_area'].median()).values.reshape(-1, 1)
        total_area_norm = scaler.fit_transform(total_area).flatten()

        rooms = df['apartment_rooms'].fillna(1).values.reshape(-1, 1)
        rooms_norm = scaler.fit_transform(rooms).flatten()

        bathrooms = df['apartment_bathrooms'].fillna(1).values.reshape(-1, 1)
        bathrooms_norm = scaler.fit_transform(bathrooms).flatten()

        # Calculate quality score (weighted average)
        quality_score = (
                0.3 * construction_year_norm +
                0.3 * total_area_norm +
                0.2 * rooms_norm +
                0.2 * bathrooms_norm
        )

        # Scale to 0-10 range with proper handling of edge cases
        quality_min = quality_score.min()
        quality_max = quality_score.max()

        # Avoid division by zero
        if quality_max - quality_min > 0:
            quality_score = (quality_score - quality_min) / (quality_max - quality_min) * 10
        else:
            # If all values are the same, set to a reasonable default
            quality_score = np.full_like(quality_score, 5.0)

        return quality_score

    except Exception as e:
        print(f"Error calculating quality score: {e}")
        # Return a default quality score if calculation fails
        return np.full(len(df), 5.0)  # Default quality score of 5/10


def recommend_properties(country=None, sort_by='price_asc', max_price=None, min_quality=None, properties_df=None):
    try:
        # Make a copy to avoid modifying the original dataframe
        filtered_properties = properties_df.copy()

        # Filter by country if specifically requested
        if country and country.lower() != 'all' and country.lower() != '':
            exact_match = filtered_properties[filtered_properties['country'].str.lower() == country.lower()]
            if len(exact_match) > 0:
                filtered_properties = exact_match
                print(f"Exact match filtered by country: {country}, found {len(filtered_properties)} properties")

        # Filter by max price if specified
        if max_price:
            before_price = len(filtered_properties)
            filtered_properties = filtered_properties[filtered_properties['price_in_USD'] <= max_price]
            print(
                f"Filtered by max price: {max_price}, found {len(filtered_properties)} properties (was {before_price})")

        # Filter by minimum quality if specified - but don't filter if it would remove all results
        if min_quality is not None:
            before_quality = len(filtered_properties)
            quality_filtered = filtered_properties[filtered_properties['quality_score'] >= min_quality]

            if len(quality_filtered) > 0:
                filtered_properties = quality_filtered
                print(
                    f"Filtered by min quality: {min_quality}, found {len(filtered_properties)} properties (was {before_quality})")
            else:
                print(
                    f"Warning: Min quality filter {min_quality} would remove all properties. Keeping original {before_quality} properties.")
                # Don't apply the filter if it would remove all results

        # If no filters applied or all filtered out, return some results
        if len(filtered_properties) == 0:
            print("No properties found after filtering, returning sample of original data")
            return properties_df.head(10).to_dict('records')

        # Sort based on user preference
        if sort_by == 'price_asc':
            filtered_properties = filtered_properties.sort_values('price_in_USD', ascending=True)
        elif sort_by == 'price_desc':
            filtered_properties = filtered_properties.sort_values('price_in_USD', ascending=False)
        elif sort_by == 'quality_asc':
            filtered_properties = filtered_properties.sort_values('quality_score', ascending=True)
        elif sort_by == 'quality_desc':
            filtered_properties = filtered_properties.sort_values('quality_score', ascending=False)
        elif sort_by == 'value':
            filtered_properties = filtered_properties.copy()
            filtered_properties['value_ratio'] = filtered_properties['quality_score'] / filtered_properties[
                'price_in_USD']
            filtered_properties = filtered_properties.sort_values('value_ratio', ascending=False)

        print(f"Returning {len(filtered_properties)} properties after filtering and sorting")
        return filtered_properties.head(50).to_dict('records')  # Limit to 50 results

    except Exception as e:
        print(f"Error in property recommendation: {e}")
        import traceback
        traceback.print_exc()
        return properties_df.head(10).to_dict('records')


@app.route('/get_property_recommendations', methods=['POST'])
def get_property_recommendations():
    try:
        data = request.get_json()
        country = data.get('country', None)
        sort_by = data.get('sort_by', 'price_asc')
        max_price = data.get('max_price', None)
        min_quality = data.get('min_quality', None)

        properties_df = load_properties()

        # Debug: Check what countries are available
        available_countries = properties_df['country'].unique().tolist()
        print(f"Available countries in dataset: {available_countries}")

        recommendations = recommend_properties(
            country=country,
            sort_by=sort_by,
            max_price=max_price,
            min_quality=min_quality,
            properties_df=properties_df
        )

        # Store recommendations in session
        session['property_recommendations'] = recommendations
        session['property_recommendation_time'] = datetime.now().isoformat()

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations),
            'available_countries': available_countries  # Send available countries to frontend
        })
    except Exception as e:
        print(f"Error in get_property_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to generate property recommendations'
        })

# Convert weight to numeric function
def convert_weight(weight_str):
    if pd.isna(weight_str):
        return 1.0
    if isinstance(weight_str, (int, float)):
        return float(weight_str)

    weight_str = str(weight_str).lower()
    match = re.search(r'(\d+\.?\d*)', weight_str)
    if match:
        numeric_value = float(match.group(1))
        if 'kg' in weight_str:
            return numeric_value
        elif 'g' in weight_str:
            return numeric_value / 1000
        elif 'l' in weight_str:
            return numeric_value
        elif 'ml' in weight_str:
            return numeric_value / 1000
        elif 'pcs' in weight_str or 'ct' in weight_str:
            return numeric_value
        else:
            return numeric_value
    else:
        return 1.0


def extract_numeric_area(area_str):
    """
    Extract numeric value from area strings like '120 m²', '500 m²', etc.
    Returns the numeric value or NaN if not convertible.
    """
    if pd.isna(area_str):
        return np.nan

    if isinstance(area_str, (int, float)):
        return float(area_str)

    # Convert to string and remove common units
    area_str = str(area_str).lower()

    # Remove common area units
    area_str = area_str.replace('m²', '').replace('m2', '').replace('sqm', '')
    area_str = area_str.replace('sq ft', '').replace('sq.ft.', '').replace('sq. ft.', '')
    area_str = area_str.replace('ft²', '').replace('ft2', '')

    # Remove any non-numeric characters except decimal point
    area_str = re.sub(r'[^\d.]', '', area_str)

    try:
        return float(area_str) if area_str else np.nan
    except ValueError:
        return np.nan

# Enhanced ML recommendation function with error handling
def train_recommendation_model(products_df):
    # Create features for ML
    product_features = products_df[['price', 'weight']].copy()
    product_features['weight_numeric'] = product_features['weight'].apply(convert_weight)

    # Select features for clustering
    X = product_features[['price', 'weight_numeric']].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and fit model
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    # Add cluster labels to products
    products_df['cluster'] = kmeans.labels_

    # Save the model and scaler for later use
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return products_df


# Load or train model with error handling
def get_trained_model(products_df):
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if os.path.exists('models/kmeans_model.pkl') and os.path.exists('models/scaler.pkl'):
        try:
            # Load pre-trained model
            kmeans = joblib.load('models/kmeans_model.pkl')
            scaler = joblib.load('models/scaler.pkl')

            # Prepare features for clustering
            product_features = products_df[['price', 'weight']].copy()
            product_features['weight_numeric'] = product_features['weight'].apply(convert_weight)
            X = product_features[['price', 'weight_numeric']].values
            X_scaled = scaler.transform(X)

            # Add cluster labels to products
            products_df['cluster'] = kmeans.predict(X_scaled)
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Training new model instead...")
            products_df = train_recommendation_model(products_df)
    else:
        # Train new model
        products_df = train_recommendation_model(products_df)

    return products_df


# Enhanced recommendation function with better error handling
def recommend_products(budget, products_df):
    try:
        # Get products with trained model
        products_df = get_trained_model(products_df)

        # Filter products within budget
        affordable_products = products_df[products_df['price'] <= budget].copy()

        if len(affordable_products) == 0:
            # If no products within budget, recommend the cheapest items
            affordable_products = products_df.nsmallest(3, 'price')
            return affordable_products.to_dict('records')

        # Calculate how many items to recommend based on budget
        avg_price = affordable_products['price'].mean()
        num_recommendations = min(10, max(3, int(budget / avg_price)))

        # Select diverse recommendations from different clusters
        recommendations = []
        clusters = affordable_products['cluster'].unique()

        for cluster in clusters:
            cluster_products = affordable_products[affordable_products['cluster'] == cluster].copy()
            if len(cluster_products) > 0:
                cluster_products.loc[:, 'weight_numeric'] = cluster_products['weight'].apply(convert_weight)
                cluster_products.loc[:, 'value_ratio'] = cluster_products['weight_numeric'] / cluster_products['price']
                best_value_product = cluster_products.nlargest(1, 'value_ratio')
                recommendations.append(best_value_product.iloc[0].to_dict())

        # If we need more recommendations, add the next best value products
        if len(recommendations) < num_recommendations:
            remaining_slots = num_recommendations - len(recommendations)
            # Get products not already recommended
            recommended_ids = [r['id'] for r in recommendations]
            remaining_products = affordable_products[~affordable_products['id'].isin(recommended_ids)].copy()

            if len(remaining_products) > 0:
                remaining_products.loc[:, 'weight_numeric'] = remaining_products['weight'].apply(convert_weight)
                remaining_products.loc[:, 'value_ratio'] = remaining_products['weight_numeric'] / remaining_products[
                    'price']
                additional_recommendations = remaining_products.nlargest(remaining_slots, 'value_ratio')
                recommendations.extend(additional_recommendations.to_dict('records'))

        return recommendations[:num_recommendations]

    except Exception as e:
        print(f"Error in recommendation: {e}")
        # Fallback: return some products
        return products_df.nsmallest(5, 'price').to_dict('records')


def recommend_products_by_category(categories, budget=None, sort_by='popularity', products_df=None):
    try:
        # Filter by categories
        if categories:
            # Convert to list if it's a string
            if isinstance(categories, str):
                categories = [categories]

            # Filter products by category
            filtered_products = products_df[products_df['category'].isin(categories)]
        else:
            filtered_products = products_df.copy()

        # Filter by budget if specified
        if budget:
            filtered_products = filtered_products[filtered_products['price'] <= budget]

        # Sort based on user preference
        if sort_by == 'popularity':
            filtered_products = filtered_products.sort_values('popularity', ascending=False)
        elif sort_by == 'price_asc':
            filtered_products = filtered_products.sort_values('price', ascending=True)
        elif sort_by == 'price_desc':
            filtered_products = filtered_products.sort_values('price', ascending=False)

        return filtered_products.to_dict('records')

    except Exception as e:
        print(f"Error in category-based recommendation: {e}")
        return products_df.nsmallest(5, 'price').to_dict('records')
# Flask routes
@app.route('/')
def index():
    return render_template('stockvel_home.html')


@app.route('/recommender')
def recommender():
    # Load categories for product recommendations
    products_df = load_products()
    categories = products_df['category'].unique().tolist()

    return render_template('recommender.html', categories=categories)


@app.route('/property_recommender')
def property_recommender():
    return render_template('property_recommender.html')


@app.route('/category_recommender')
def category_recommender():
    products_df = load_products()
    categories = products_df['category'].unique().tolist()
    return render_template('category_recommender.html', categories=categories)


@app.route('/search')
def search():
    query = request.args.get('q', '')
    products_df = load_products()

    if query:
        # Simple search by name
        results = products_df[products_df['name'].str.contains(query, case=False)]
    else:
        results = pd.DataFrame()

    return render_template('search.html', results=results.to_dict('records'), query=query)


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        budget = float(data.get('budget', 0))
        recommendation_type = data.get('type', 'products')
        categories = data.get('categories', None)

        if recommendation_type == 'products':
            products_df = load_products()

            if categories:
                # Use category-based recommendation
                recommendations = recommend_products_by_category(
                    categories=categories,
                    budget=budget,
                    products_df=products_df
                )
            else:
                # Use budget-based recommendation
                recommendations = recommend_products(budget, products_df)

            # Store recommendations in session for wishlist functionality
            session['recommendations'] = recommendations
            session['recommendation_time'] = datetime.now().isoformat()

            return jsonify({
                'success': True,
                'recommendations': recommendations,
                'count': len(recommendations),
                'type': 'products'
            })
        else:
            # This should not happen as property recommendations have their own endpoint
            return jsonify({
                'success': False,
                'error': 'Invalid recommendation type'
            })

    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate recommendations'
        })


@app.route('/debug_property_columns')
def debug_property_columns():
    """Debug route to check property data column details"""
    try:
        df = pd.read_csv('data/world_real_estate_data(147k).csv', nrows=10)  # Read just first 10 rows

        debug_info = {
            'columns': df.columns.tolist(),
            'first_few_rows': df.head().to_dict('records'),
            'country_values': df['country'].unique().tolist() if 'country' in df.columns else 'No country column',
            'price_column_values': df[
                'price_in_USD'].head().tolist() if 'price_in_USD' in df.columns else 'No price_in_USD column'
        }

        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug_properties')
def debug_properties():
    """Debug route to check property data loading"""
    properties_df = load_properties()

    debug_info = {
        'total_properties': len(properties_df),
        'available_countries': properties_df['country'].unique().tolist(),
        'country_counts': properties_df['country'].value_counts().to_dict(),
        'columns': properties_df.columns.tolist(),
        'first_few_rows': properties_df.head().to_dict('records')
    }

    return jsonify(debug_info)


# Add these error handlers to your Flask app
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # Log the error
    print(f"Unhandled exception: {e}")

    # Return JSON response for API errors
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(e)
    }), 500


@app.route('/debug_property_response')
def debug_property_response():
    """Debug route to test property response format"""
    properties_df = load_properties()
    recommendations = recommend_properties(
        country='Turkey',
        sort_by='price_asc',
        max_price=300000,
        min_quality=7.0,
        properties_df=properties_df
    )

    # Check the first recommendation to see the structure
    if recommendations:
        sample_property = recommendations[0]
        print("Sample property structure:", sample_property.keys())
        print("Price field exists:", 'price_in_USD' in sample_property)
        print("Price value:", sample_property.get('price_in_USD', 'NOT FOUND'))

    return jsonify({
        'sample_property': recommendations[0] if recommendations else {},
        'all_keys': list(recommendations[0].keys()) if recommendations else []
    })

@app.route('/save_wishlist', methods=['POST'])
def save_wishlist():
    try:
        wishlist = request.json.get('wishlist', [])

        # Create PDF with better formatting
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 800, "Your Stockvel Wishlist")
        p.setFont("Helvetica", 12)
        p.drawString(100, 780, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        y = 750
        total = 0
        for item in wishlist:
            if y < 50:  # Add new page if running out of space
                p.showPage()
                p.setFont("Helvetica", 12)
                y = 800

            p.drawString(100, y, f"{item['name']}")
            p.drawString(400, y, f"${item['price']}")
            p.drawString(500, y, f"{item.get('weight', 'N/A')}")
            y -= 20
            total += item['price']

        # Add total
        y -= 20
        p.line(100, y, 550, y)
        y -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y, "Total:")
        p.drawString(400, y, f"${total:.2f}")

        p.showPage()
        p.save()

        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='stockvel_wishlist.pdf', mimetype='application/pdf')
    except Exception as e:
        print(f"Error saving wishlist: {e}")
        return jsonify({"error": "Failed to save wishlist"}), 500


@app.route('/share_wishlist', methods=['POST'])
def share_wishlist():
    try:
        data = request.get_json()
        wishlist = data.get('wishlist', [])
        method = data.get('method', 'email')

        # In a real implementation, you would integrate with email/WhatsApp/SMS APIs here
        # For now, we'll just return a success message

        total = sum(item['price'] for item in wishlist)
        item_count = len(wishlist)

        return jsonify({
            'success': True,
            'message': f'Wishlist with {item_count} items (Total: ${total:.2f}) ready to share via {method}',
            'method': method
        })
    except Exception as e:
        print(f"Error sharing wishlist: {e}")
        return jsonify({"error": "Failed to share wishlist"}), 500


@app.route('/wishlist')
def wishlist():
    return render_template('wishlist.html')



# Load environment variables
load_dotenv()


# Add this route to your main.py (place it with your other routes)
@app.route('/send_wishlist_sms', methods=['POST'])
def send_wishlist_sms():
    try:
        data = request.get_json()
        wishlist = data.get('wishlist', [])
        phone_number = data.get('phone_number', '')

        if not wishlist:
            return jsonify({
                'success': False,
                'error': 'Wishlist is empty'
            })

        if not phone_number:
            return jsonify({
                'success': False,
                'error': 'Phone number is required'
            })

        # Validate phone number format
        if not re.match(r'^\+?[1-9]\d{1,14}$', phone_number):
            return jsonify({
                'success': False,
                'error': 'Please enter a valid phone number with country code'
            })

        # Format the wishlist message (SMS has 160 character limit)
        total = sum(item['price'] for item in wishlist)
        item_count = len(wishlist)

        # Base message
        message = f"Stockvel Wishlist: {item_count} items, Total: ${total:.2f}. "

        # Add top items if there's space
        remaining_chars = 160 - len(message) - 3  # Reserve space for ellipsis
        if remaining_chars > 20:  # Only add items if we have reasonable space
            items_text = ""
            for i, item in enumerate(wishlist[:3]):  # Max 3 items
                item_str = f"{item['name']} (${item['price']:.2f})"
                if len(items_text + item_str) > remaining_chars:
                    items_text += "..."
                    break
                if i > 0:
                    items_text += ", "
                items_text += item_str

            message += items_text

        # Send SMS using Infobip API
        conn = http.client.HTTPSConnection("api.infobip.com")
        payload = json.dumps({
            "messages": [
                {
                    "destinations": [{"to": phone_number}],
                    "from": "447491163443",
                    "text": message
                }
            ]
        })

        api_key ='322635f06aff17eddffa20a7b14a0753-67487056-c479-42f5-bc6f-c669227e65e4'
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'SMS service not configured'
            })

        headers = {
            'Authorization': f'App {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        conn.request("POST", "/sms/2/text/advanced", payload, headers)
        res = conn.getresponse()
        response_data = res.read().decode("utf-8")

        if res.status == 200:
            print(f"SMS sent successfully to {phone_number}")
            return jsonify({
                'success': True,
                'message': f'Wishlist sent via SMS to {phone_number}'
            })
        else:
            print(f"SMS failed: {response_data}")
            return jsonify({
                'success': False,
                'error': 'Failed to send SMS. Please check the phone number.'
            })

    except Exception as e:
        print(f"Error sending SMS: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to send SMS. Please try again.'
        })

# Mpilo AI Chatbot Configuration (keep your existing chatbot code here)
CHATBOT_SYSTEM_PROMPT = """
You are Mpilo, an AI chatbot for SmartShopper, a smart shopping recommendation website. 

Your role:
- Help users understand how to use SmartShopper features
- Provide guidance on recommendations, wishlist, budget settings
- Answer questions about the website functionality
- Be friendly, helpful, and concise
- Always introduce yourself as Mpilo

SmartShopper Features:
1. RECOMMENDATIONS: Users can get AI-powered product recommendations based on their budget
2. WISHLIST: Users can save products to a wishlist and download as PDF
3. BUDGET: Users can set and manage their shopping budget
4. SEARCH: Users can search for specific products
5. CHAT: Mpilo (you) helps users navigate the website

Guidelines:
- Always be helpful and encouraging
- Provide step-by-step instructions when needed
- If you don't know something specific about SmartShopper, say so politely
- Keep responses concise but informative
- Use emojis occasionally to make conversations friendly
- Always end with asking if there's anything else you can help with
- Remember your name is Mpilo
"""

# Global chatbot instance
conversational_ai = None


def load_huggingface_model():
    """Load Hugging Face conversational model"""
    global conversational_ai
    try:
        print("Loading Mpilo AI model...")

        # Using a smaller, faster model for web deployment
        model_name = "microsoft/DialoGPT-small"  # Smaller model for faster loading

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create conversation pipeline
        conversational_ai = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        print("Mpilo AI model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading Mpilo AI model: {e}")
        print("Falling back to rule-based responses...")
        return False


def get_hf_chatbot_response(user_message, conversation_history=None):
    """Get response using Hugging Face transformers"""
    global conversational_ai

    try:
        if conversational_ai is None:
            return "I'm Mpilo, and I'm currently unavailable. Please try again later."

        # Prepare conversation context
        context = "Mpilo: Hello! I'm Mpilo, your SmartShopper AI assistant. "

        if conversation_history:
            # Build context from recent conversation
            for msg in conversation_history[-4:]:  # Keep last 4 messages for context
                if msg['role'] == 'user':
                    context += f"User: {msg['content']} "
                elif msg['role'] == 'assistant':
                    context += f"Mpilo: {msg['content']} "

        # Add current message
        context += f"User: {user_message} Mpilo:"

        # Generate response
        response = conversational_ai(
            context,
            max_length=len(context.split()) + 30,  # Limit response length
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=conversational_ai.tokenizer.eos_token_id
        )

        # Extract the generated response
        generated_text = response[0]['generated_text']

        # Extract only Mpilo's response
        if "Mpilo:" in generated_text:
            mpilo_response = generated_text.split("Mpilo:")[-1].strip()
        else:
            mpilo_response = generated_text.split("User:")[-1].strip()

        # Clean up the response
        mpilo_response = mpilo_response.replace("User:", "").strip()

        # If response is too short or doesn't make sense, provide a fallback
        if len(mpilo_response) < 10 or "User:" in mpilo_response:
            return get_fallback_response(user_message)

        return mpilo_response

    except Exception as e:
        print(f"Error in Mpilo chatbot: {e}")
        return get_fallback_response(user_message)


def get_fallback_response(user_message):
    """Fallback responses when AI model fails"""
    user_message = user_message.lower().strip()

    # Simple keyword-based responses
    if any(word in user_message for word in ["recommend", "recommendation", "suggest"]):
        return "Hi! I'm Mpilo, your SmartShopper assistant. To get recommendations, go to the 'Recommend' page, enter your budget, and click 'Get Recommendations'. Our AI will suggest the best products for your budget! 🤖"

    elif any(word in user_message for word in ["budget", "money", "price", "cost"]):
        return "Hello! I'm Mpilo. You can set your budget using the 'Set Budget' button in the navigation bar. The system will help you stay within your spending limit! 💰"

    elif any(word in user_message for word in ["wishlist", "save", "favorite"]):
        return "Hi there! I'm Mpilo. You can add items to your wishlist by clicking 'Add to Wishlist' on any product. View your wishlist by clicking the heart icon! ❤️"

    elif any(word in user_message for word in ["search", "find", "look for"]):
        return "Hello! I'm Mpilo. Use the Search page to find specific products. You can search by product name or browse categories! 🔍"

    elif any(word in user_message for word in ["property", "real estate", "house", "apartment"]):
        return "Hello! I'm Mpilo. We now offer property recommendations! Visit the Property Recommender page to find properties based on country, price, and quality. 🏠"

    elif any(word in user_message for word in ["category", "categories"]):
        return "Hi! I'm Mpilo. You can now get recommendations by product categories! Visit the Category Recommender page to select multiple categories and get popular products. 🛍️"

    elif any(word in user_message for word in ["hello", "hi", "hey"]):
        return "Hello! I'm Mpilo, your SmartShopper AI assistant. I can help you with recommendations, wishlist, budget settings, and more! How can I assist you today? 🤖"

    elif any(word in user_message for word in ["help", "support"]):
        return "I'm Mpilo, and I'm here to help! You can ask me about recommendations, wishlist management, budget settings, or any other SmartShopper features. What would you like to know? 🤝"

    elif any(word in user_message for word in ["name", "who are you", "what are you"]):
        return "I'm Mpilo! I'm your AI assistant for SmartShopper. I help users navigate the website, understand features, and get the most out of their shopping experience. How can I help you today? 😊"

    else:
        return "Hi! I'm Mpilo, your SmartShopper AI assistant! You can ask me about recommendations, wishlist, budget settings, or how to use any features. What would you like to know? 😊"


# Store conversation history in session
@app.route('/chat_history', methods=['POST'])
def save_chat_history():
    """Save conversation history"""
    try:
        data = request.get_json()
        history = data.get('history', [])
        session['chat_history'] = history
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/mpilo_chatbot', methods=['POST'])
def mpilo_chatbot():
    """Handle Mpilo AI chatbot messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])

        if not user_message:
            return jsonify({
                'success': False,
                'response': 'Please enter a message.'
            })

        # Get AI response
        ai_response = get_hf_chatbot_response(user_message, conversation_history)

        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in Mpilo chatbot: {e}")
        return jsonify({
            'success': False,
            'response': 'I apologize, but I encountered an error. Please try again.'
        })


@app.route('/mpilo_chat')
def mpilo_chat():
    """Mpilo AI Chatbot page"""
    return render_template('chatbot.html')


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/property', exist_ok=True)

    app.run(debug=True)