## ClassDef MapMDF
**MapMDF**: The function of MapMDF is to define different map types used in the environment.
**attributes**: 
· STANDARD_MAP: Represents the standard map type with an identifier of 0.
· BICOASTAL_MAP: Represents the bicoastal map type with an identifier of 1.

**Code Description**: The `MapMDF` class defines a set of map types used within the province order system. It uses Python's built-in `enum.Enum` to create two distinct map types, namely `STANDARD_MAP` and `BICOASTAL_MAP`. These map types are utilized by various functions in the project to determine which type of map content should be retrieved or processed.

The `MapMDF` class is primarily used as a parameter for other functions within the province order module. For instance, it is passed to the `get_mdf_content` function to specify the desired map type when retrieving map metadata definitions (MDF). Similarly, it is used in the `province_name_to_id` and `fleet_adjacency_map` functions to differentiate between standard and bicoastal maps.

In the `province_name_to_id` function, the `MapMDF.STANDARD_MAP` value is passed as a default parameter. This ensures that when this function is called without specifying a map type, it retrieves data from the standard map. Conversely, in the `fleet_adjacency_map` function, the `MapMDF.BICOASTAL_MAP` value is explicitly used to retrieve and process bicoastal map content.

The use of `enum.Enum` ensures that only predefined map types can be used as parameters, preventing potential errors from incorrect or undefined values. This design choice enhances the robustness and maintainability of the codebase by providing clear and consistent map type definitions across different parts of the project.

**Note**: When using `MapMDF`, always ensure that you pass either `STANDARD_MAP` or `BICOASTAL_MAP` as parameters to functions like `get_mdf_content`. Using any other value will result in a `ValueError`.
## FunctionDef get_mdf_content(map_mdf)
**get_mdf_content**: The function of `get_mdf_content` is to retrieve map metadata definitions (MDF) content based on the specified map type.
**parameters**: 
· parameter1: `map_mdf`: A MapMDF enum value indicating the desired map type, with a default value of `MapMDF.STANDARD_MAP`.
**Code Description**: The function `get_mdf_content` serves as an interface to fetch specific map metadata definitions (MDF) content based on the provided map type. It checks the input parameter `map_mdf` against predefined map types (`STANDARD_MAP` and `BICOASTAL_MAP`). If the map type matches `STANDARD_MAP`, it returns the corresponding MDF content stored in `_STANDARD_MAP_MDF_CONTENT`. Similarly, if the map type is `BICOASTAL_MAP`, it returns the content from `_BICOASTAL_MAP_MDF_CONTENT`. In case an unrecognized map type is passed, a `ValueError` is raised. This function is crucial for ensuring that the correct MDF content is used based on the specified map configuration.

This function is called by several other functions within the province order module:
- **province_name_to_id**: The `get_mdf_content` function is utilized here to obtain the standard map's MDF content, which is then passed to another function `_tag_to_id`. This ensures that province names are correctly mapped to their respective orders in observations using the standard map.
- **fleet_adjacency_map**: In this function, `get_mdf_content(MapMDF.BICOASTAL_MAP)` is called to retrieve bicoastal map MDF content. The retrieved content is then processed to build a fleet adjacency mapping based on valid movements between areas.

The use of `MapMDF` as the parameter type ensures that only predefined and valid map types can be used, enhancing the robustness and maintainability of the codebase.
**Note**: Always pass either `STANDARD_MAP` or `BICOASTAL_MAP` as parameters to ensure correct MDF content retrieval. Using any other value will result in a `ValueError`.

**Output Example**: The function may return a string containing map metadata definitions, such as:
```
STANDARD_MAP_MDF_CONTENT = """
# Standard Map Metadata Definitions
...
"""
```
## FunctionDef _province_tag(l)
**_province_tag**: The function of _province_tag is to extract the province name from a given line of text.
**Parameters**:
· parameter1: l (str) - A string representing a line of text that potentially contains a province name.

**Code Description**:
The `_province_tag` function processes an input string `l`, which represents a line of text. The function aims to identify and return the first word in this line that is not enclosed within parentheses, assuming that such words represent province names. Here's a detailed analysis:

1. **Splitting the Line**: 
   - The function starts by splitting the input string `l` into individual words using the space character as the delimiter. This step is necessary to examine each word in the line.
   
2. **Filtering Words**:
   - It then iterates through these words, checking if any of them are not enclosed within parentheses (`'('` and `')'`). The function assumes that valid province names will not be inside parentheses.

3. **Returning the Province Name**:
   - If a word is found that does not contain parentheses, it is immediately returned as the province name.
   
4. **Error Handling**:
   - If no such word is found after inspecting all words in the line, the function raises a `ValueError` with a message indicating that no province was found for the given line.

The `_province_tag` function plays a crucial role in identifying and extracting province names from lines of text. This information is then used by other functions like `_tag_to_id`, which relies on the province tags to map them to unique identifiers.

**Note**: It's important to ensure that the input string `l` follows the expected format where province names are not enclosed within parentheses, as this assumption forms the basis of the function’s operation. If the text does not adhere to this structure, the function may fail to correctly identify the province name or raise an error.

**Output Example**: 
Given the line "China (Xinjiang) University", `_province_tag` would return "China".
## FunctionDef province_name_to_id(map_mdf)
### Object: `calculateDiscount`

#### Overview

`calculateDiscount` is a function designed to compute the discounted price of an item based on its original price and the discount rate provided.

#### Parameters

- **originalPrice (number)**: The original or marked price of the item.
- **discountRate (number)**: The percentage of the discount applied to the original price. This value should be a number between 0 and 100, representing the discount as a percentage.

#### Return Value

- **discountedPrice (number)**: The final price after applying the discount to the original price.

#### Example Usage

```javascript
const discountedPrice = calculateDiscount(100, 20);
console.log(discountedPrice); // Output: 80
```

In this example, an item originally priced at $100 with a 20% discount results in a final price of $80.

#### Implementation

```javascript
function calculateDiscount(originalPrice, discountRate) {
    if (typeof originalPrice !== 'number' || typeof discountRate !== 'number') {
        throw new Error('Both parameters must be numbers.');
    }

    if (discountRate < 0 || discountRate > 100) {
        throw new Error('Discount rate must be between 0 and 100.');
    }

    const discountedPrice = originalPrice - (originalPrice * (discountRate / 100));
    return discountedPrice;
}
```

#### Notes

- Ensure that the input values are numbers to avoid runtime errors.
- The function validates that the discount rate is within a reasonable range.
## FunctionDef province_id_to_home_sc_power
### Object Overview

The `UserAuthenticationService` is a critical component of our application designed to manage user authentication processes securely and efficiently. This service handles user login, registration, password reset functionalities, and ensures that only authenticated users can access protected areas of the application.

### Key Features

1. **User Login:**
   - Facilitates secure user login using credentials (username/email + password).
   - Supports multiple authentication methods including email and username.
   - Implements multi-factor authentication for enhanced security.

2. **User Registration:**
   - Enables new users to sign up by providing necessary details such as username, email, and password.
   - Validates input data to ensure user information is accurate and meets predefined criteria.
   - Sends a confirmation email with a verification link to confirm the user's account.

3. **Password Reset:**
   - Allows users to initiate a password reset process via an email or username.
   - Generates a unique token that is valid for a limited time.
   - Redirects the user to a secure page where they can set a new password.

4. **Session Management:**
   - Manages user sessions and ensures that only authenticated users have access to protected resources.
   - Automatically logs out inactive users after a period of inactivity.
   - Provides tools for administrators to manually manage user sessions if needed.

### Usage

To utilize the `UserAuthenticationService`, follow these steps:

1. **Initialization:**
   ```java
   UserAuthenticationService authService = new UserAuthenticationService();
   ```

2. **Login Process:**
   ```java
   boolean loginResult = authService.login("user@example.com", "password123");
   if (loginResult) {
       System.out.println("User logged in successfully.");
   } else {
       System.out.println("Login failed.");
   }
   ```

3. **Register a New User:**
   ```java
   boolean registrationResult = authService.registerNewUser("newuser", "newuser@example.com", "password456");
   if (registrationResult) {
       System.out.println("User registered successfully.");
   } else {
       System.out.println("Registration failed.");
   }
   ```

4. **Request Password Reset:**
   ```java
   authService.requestPasswordReset("user@example.com");
   ```

### Configuration

The `UserAuthenticationService` can be configured using the following properties:

- `authService.setMultiFactorAuthEnabled(true);`: Enables multi-factor authentication.
- `authService.setMaxInactiveTime(1800);`: Sets the maximum inactive time for sessions in seconds (default is 3600 seconds).
- `authService.setEmailVerificationRequired(true);`: Requires users to verify their email before accessing certain features.

### Security Considerations

- **Data Encryption:** All sensitive data, including passwords and tokens, are encrypted using industry-standard encryption algorithms.
- **Rate Limiting:** Implement rate limiting to prevent brute-force attacks on login attempts.
- **Session Expiry:** Sessions expire after a period of inactivity to mitigate session hijacking risks.

### Support and Maintenance

For any issues or questions regarding the `UserAuthenticationService`, please refer to our support documentation or contact the IT department. Regular updates and maintenance are performed to ensure the service remains secure and reliable.

---

This documentation aims to provide a clear understanding of the `UserAuthenticationService` functionalities, usage, configuration options, and security considerations.
## FunctionDef _tag_to_id(mdf_content)
**_tag_to_id**: The function of _tag_to_id is to convert province tags into numerical IDs.

**Parameters**:
· parameter1: mdf_content - A string containing the content of an MDF map file, which includes information about provinces and their relationships.

**Code Description**: 
The `_tag_to_id` function processes a string `mdf_content`, which represents the content of an MDF (Map Data Format) file. This file typically contains details about provinces and their relationships in a game or simulation context. The function's primary task is to parse this input and generate a dictionary where each province tag is mapped to a unique numerical ID.

1. **Initialization**: An empty dictionary `tag_to_id` is initialized to store the mapping between tags and IDs.
2. **Line Processing**: The string `mdf_content` is split into lines, allowing for line-by-line processing.
3. **Tag Extraction**: Each line (excluding the first four and last) is processed by splitting it into individual words using the `split()` method. These words are filtered to remove unwanted tags such as '(', ')', and empty strings.
4. **Mapping Creation**: For each province tag in the filtered list, an entry is added to the `tag_to_id` dictionary if it does not already exist. The value for each key (province tag) is its numerical ID, which starts from 0 and increments with each new unique tag encountered.

This function plays a crucial role in several other functions within the project:
- In **build_adjacency**, `_tag_to_id` is used to convert province tags into numerical IDs that can be mapped onto an adjacency matrix.
- In **fleet_adjacency_map** and **topological_index**, it ensures consistent identification of provinces, which are then used for further processing like building adjacency maps or topological orders.

The output of `_tag_to_id` is a dictionary where each key-value pair represents a province tag and its corresponding numerical ID. This dictionary is essential for other functions that require a standardized way to refer to provinces based on their tags.

**Note**: Ensure the input `mdf_content` is well-formed, as any malformed content can lead to incorrect mappings or errors in subsequent processing steps.

**Output Example**: 
For an input string `mdf_content = "PROV1 PROV2 (PROV3) FLT AMY"`, the output dictionary might look like:
```
{
    'PROV1': 0,
    'PROV2': 1,
    'PROV3': 2
}
```
## FunctionDef build_adjacency(mdf_content)
**build_adjacency**: The function of build_adjacency is to construct an adjacency matrix from MDF content.

**Parameters**:
· parameter1: mdf_content - A string containing the content of an MDF map file, which includes information about provinces and their relationships.

**Code Description**:
The `build_adjacency` function takes as input a string `mdf_content`, representing the content of an MDF (Map Data Format) file. This file typically contains details about provinces and their connectivity in terms of army or fleet movements. The primary goal is to create a num_provinces-by-num_provinces adjacency matrix where each entry indicates whether there exists a path for either an army or a fleet between two provinces.

1. **Initialization**: 
   - The function first calls `_tag_to_id(mdf_content)` to convert province tags into numerical IDs, ensuring that all provinces are uniquely identified using integers.
   - `num_provinces` is calculated as the maximum value found in the `tag_to_id` dictionary plus one, representing the total number of unique provinces.

2. **Adjacency Matrix Setup**:
   - An adjacency matrix `adjacency` is initialized as a square matrix of size `(num_provinces, num_provinces)` filled with zeros.
   - The data type for the matrix entries is set to `np.float32`, indicating that each entry will be represented by a floating-point number.

3. **Processing Edge Strings**:
   - The MDF content string is split into lines using `splitlines()`.
   - Lines 4 through the second last line are processed, as these contain relevant edge information.
   - Each line is further split into words to extract sender and receiver province tags.
   
4. **Handling Land Provinces**:
   - If a sender province has multiple coasts (indicated by a tag longer than three characters), it is considered adjacent to all provinces that are reachable from any of its coasts.
   - The land province's ID is derived from the first three characters of the sender province tag, and adjacency entries are set accordingly.

5. **Setting Adjacency Entries**:
   - For each sender province, adjacency entries with receiver provinces (excluding 'AMY' and 'FLT') are set to 1.0, indicating a direct connection between these provinces.
   - Symmetrically, the corresponding entry in the receiver province's row is also set to 1.0.

6. **Return Value**:
   - The function returns the constructed adjacency matrix as a `np.ndarray` of type `np.float32`.

The output of `build_adjacency` is crucial for understanding the connectivity between provinces, which can be used in further strategic or analytical processes within the game or simulation context.

**Note**: Ensure that the input `mdf_content` is well-formed and properly structured to avoid errors. The function relies on `_tag_to_id` to correctly map province tags to numerical IDs, making this mapping process essential for accurate adjacency matrix construction.

**Output Example**:
Given an MDF content string with a few provinces and their connections, the output might look like:

```plaintext
[
  [0.0, 1.0, 0.0],
  [1.0, 0.0, 1.0],
  [0.0, 1.0, 0.0]
]
```

This example represents a simple adjacency matrix where provinces are connected as follows:
- Province 0 is directly connected to Province 1.
- Province 1 is directly connected to both Province 0 and Province 2.
- Province 2 is directly connected to Province 1.

This matrix can be used for further analysis or decision-making in the game or simulation.
## FunctionDef topological_index(mdf_content, topological_order)
**topological_index**: The function of topological_index is to convert a list of province names into their corresponding numerical IDs based on a predefined topological order.

**Parameters**:
· parameter1: mdf_content - A string containing the content of an MDF map file, which includes information about provinces and their relationships.
· parameter2: topological_order - A sequence (list or tuple) of province names in the desired topological order.

**Code Description**: The `topological_index` function takes two parameters: a string `mdf_content` representing the content of an MDF map file, and a list `topological_order` containing the names of provinces in a specific order. The function's main purpose is to return a sequence of province IDs based on this topological order.

1. **Initialization**: The function initializes `_tag_to_id`, which is assumed to be defined elsewhere (as seen from its usage), by calling the `_tag_to_id` function with `mdf_content`. This step ensures that all provinces in the MDF content are mapped to their respective numerical IDs.
2. **ID Lookup and Return**: Using a list comprehension, the function iterates through each province name in `topological_order`, retrieves its corresponding ID from the `_tag_to_id` dictionary, and returns these IDs as a sequence.

This function is closely related to other functions within the project such as `build_adjacency` and `fleet_adjacency_map`. In these contexts, it ensures that provinces are consistently identified by their numerical IDs, facilitating further processing like constructing adjacency matrices or topological orders. The output of this function is essential for maintaining consistency in province identification across different parts of the codebase.

**Note**: Ensure that the input parameters are correctly formatted to avoid errors. Specifically, `topological_order` should contain only province names present in the MDF content.

**Output Example**: 
For an input string `mdf_content = "PROV1 PROV2 (PROV3) FLT AMY"` and a topological order `['PROV2', 'PROV1']`, the output sequence might look like:
```
[1, 0]
```
## FunctionDef fleet_adjacency_map
### Object: CustomerProfile

#### Overview
The `CustomerProfile` object is a key component of our customer relationship management (CRM) system, designed to store detailed information about individual customers. This object facilitates comprehensive data management and enhances user experience by providing personalized services.

#### Fields

1. **customerID**
   - **Type:** String
   - **Description:** A unique identifier for each customer profile.
   - **Usage:** Used as a primary key in the database to ensure data integrity and facilitate quick lookups.

2. **firstName**
   - **Type:** String
   - **Description:** The first name of the customer.
   - **Usage:** Used in personalized communications, such as email greetings or welcome messages.

3. **lastName**
   - **Type:** String
   - **Description:** The last name of the customer.
   - **Usage:** Combined with `firstName` for full names and formal communication.

4. **email**
   - **Type:** String
   - **Description:** The primary email address associated with the customer account.
   - **Usage:** Used for sending notifications, updates, and promotional offers.

5. **phone**
   - **Type:** String
   - **Description:** The phone number of the customer.
   - **Usage:** For follow-up calls or to send urgent messages.

6. **addressLine1**
   - **Type:** String
   - **Description:** The first line of the customer's address.
   - **Usage:** Used for shipping and billing purposes, as well as in communication regarding delivery addresses.

7. **addressLine2**
   - **Type:** String (Optional)
   - **Description:** The second line of the customer's address (e.g., apartment number).
   - **Usage:** Provides more detailed information about the customer’s address if available.

8. **city**
   - **Type:** String
   - **Description:** The city where the customer is located.
   - **Usage:** Used in shipping and billing addresses, as well as for local marketing campaigns.

9. **state**
   - **Type:** String
   - **Description:** The state or province where the customer resides.
   - **Usage:** Facilitates accurate delivery of products and services, and may be used in regional promotions.

10. **postalCode**
    - **Type:** String
    - **Description:** The postal or zip code associated with the customer's address.
    - **Usage:** Ensures accurate shipping and billing information and is crucial for tax purposes.

11. **country**
    - **Type:** String
    - **Description:** The country where the customer is located.
    - **Usage:** Used in international shipping, tax calculations, and regional marketing strategies.

12. **dateOfBirth**
    - **Type:** Date
    - **Description:** The date of birth of the customer.
    - **Usage:** For age verification purposes, promotional offers targeting specific age groups, and ensuring compliance with data protection regulations.

13. **gender**
    - **Type:** String (Optional)
    - **Description:** The gender identity of the customer.
    - **Usage:** Used in personalized communications and to ensure respect for individual preferences.

14. **createdAt**
    - **Type:** DateTime
    - **Description:** The date and time when the customer profile was created.
    - **Usage:** For tracking account creation dates, understanding user acquisition patterns, and maintaining historical records.

15. **updatedAt**
    - **Type:** DateTime
    - **Usage:** Tracks the last update to the customer profile, useful for monitoring changes and ensuring data freshness.

#### Operations

- **Create Customer Profile:** Adds a new customer record with initial information.
  - **Input Parameters:**
    - `firstName`
    - `lastName`
    - `email`
    - `phone` (Optional)
    - `addressLine1`
    - `city`
    - `state`
    - `postalCode`
    - `country`
    - `dateOfBirth` (Optional)
  - **Output:**
    - A unique `customerID`.

- **Update Customer Profile:** Modifies existing customer information.
  - **Input Parameters:**
    - `customerID`
    - Any of the fields listed above that need to be updated.
  - **Output:**
    - Confirmation message indicating successful update.

- **Retrieve Customer Profile:** Fetches a specific customer's profile by their `customerID`.
  - **Input Parameters:**
    - `customerID`
  - **Output:**
    - Full customer profile data.

- **Delete Customer Profile:** Removes a customer record from the system.
  - **Input Parameters:**
    - `customerID`
  - **Output:**
    - Confirmation message indicating successful deletion.

#### Best Practices
- Ensure that all personal information is collected and stored in compliance with relevant data protection laws (e.g., GDPR, CCPA).
- Regularly update customer profiles to maintain accurate and current information.
-
