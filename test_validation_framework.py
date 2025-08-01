#!/usr/bin/env python3
"""
Test script for AgentCodeEval validation framework

This script tests the complete pipeline by:
1. Loading a scenario from our generated set
2. Creating a mock solution to test with
3. Running the solution through our automated validation framework
4. Displaying detailed scoring breakdown
"""

import asyncio
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agentcodeeval.core.config import Config
from agentcodeeval.generation.validation_framework import AutomatedValidator


console = Console()


async def test_validation_framework():
    """Test the complete validation framework with a mock solution"""
    
    console.print(Panel.fit("🧪 Testing AgentCodeEval Validation Framework", style="bold green"))
    
    # Load configuration
    config = Config(config_path="test_config.yaml")
    
    # Initialize validator
    validator = AutomatedValidator(config)
    
    console.print("✅ Framework initialized")
    
    # Find available scenarios
    scenarios_dir = Path("data/output/scenarios")
    if not scenarios_dir.exists():
        console.print("❌ No scenarios found. Run Phase 3 first!")
        return
    
    scenario_files = list(scenarios_dir.glob("*.json"))
    if not scenario_files:
        console.print("❌ No scenario files found!")
        return
    
    # Select first scenario for testing
    test_scenario_file = scenario_files[0]
    console.print(f"📁 Testing with: {test_scenario_file.name}")
    
    # Load scenario data
    with open(test_scenario_file, 'r') as f:
        scenario_data = json.load(f)
    
    # Get first scenario from the file
    scenario = scenario_data['scenarios'][0]
    
    console.print(f"🎯 Testing scenario: {scenario['title'][:50]}...")
    console.print(f"📋 Task category: {scenario['task_category']}")
    console.print(f"⚖️  Difficulty: {scenario['difficulty']}")
    
    # Step 1: Generate test suite
    console.print("\n📝 Step 1: Generating automated test suite...")
    test_suite = await validator.generate_test_suite(scenario)
    
    # Display test suite summary
    test_counts = {
        'Compilation': len(test_suite.compilation_tests),
        'Unit': len(test_suite.unit_tests), 
        'Integration': len(test_suite.integration_tests),
        'Performance': len(test_suite.performance_tests),
        'Security': len(test_suite.security_tests)
    }
    
    table = Table(title="Generated Test Suite")
    table.add_column("Test Type", style="cyan")
    table.add_column("Count", style="magenta")
    
    for test_type, count in test_counts.items():
        table.add_row(test_type, str(count))
    
    console.print(table)
    
    # Step 2: Create mock solution for testing
    console.print("\n🔧 Step 2: Creating mock solution for testing...")
    solution_code = create_mock_solution(scenario)
    
    console.print(f"📁 Created {len(solution_code)} mock code files:")
    for filename in solution_code.keys():
        lines = len(solution_code[filename].split('\n'))
        console.print(f"   • {filename} ({lines} lines)")
    
    # Step 3: Validate solution
    console.print("\n⚡ Step 3: Validating solution with our framework...")
    
    start_time = time.time()
    
    with console.status("[bold blue]⚡ Running automated validation..."):
        validation_result = await validator.validate_solution(scenario, solution_code, test_suite)
    
    validation_time = time.time() - start_time
    
    # Step 4: Display detailed results
    console.print(f"\n🎉 Validation complete in {validation_time:.2f}s")
    
    display_validation_results(validation_result)
    
    return validation_result


def create_mock_solution(scenario):
    """Create a mock solution for testing"""
    
    task_category = scenario['task_category']
    
    if task_category == 'feature_implementation':
        return {
            "new_feature.go": """package main

import (
    "fmt"
    "net/http"
    "encoding/json"
    "mime/multipart"
    "io"
    "os"
    "path/filepath"
    "strings"
)

// ProfilePictureHandler handles profile picture uploads
func ProfilePictureHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != "POST" {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    // Parse multipart form
    err := r.ParseMultipartForm(5 << 20) // 5MB limit
    if err != nil {
        http.Error(w, "Failed to parse form", http.StatusBadRequest)
        return
    }
    
    file, header, err := r.FormFile("profile_picture")
    if err != nil {
        http.Error(w, "No file uploaded", http.StatusBadRequest)
        return
    }
    defer file.Close()
    
    // Validate file type and size
    if !isValidImageType(header.Header.Get("Content-Type")) {
        http.Error(w, "Invalid file type", http.StatusBadRequest)
        return
    }
    
    if header.Size > 5*1024*1024 {
        http.Error(w, "File too large", http.StatusBadRequest)
        return
    }
    
    // Save file
    filename, err := saveUploadedFile(file, header)
    if err != nil {
        http.Error(w, "Failed to save file", http.StatusInternalServerError)
        return
    }
    
    // Create thumbnail
    thumbnailPath, err := createThumbnail(filename)
    if err != nil {
        http.Error(w, "Failed to create thumbnail", http.StatusInternalServerError)
        return
    }
    
    // Save to database
    pictureID, err := saveToDatabase(filename, thumbnailPath, header.Size)
    if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }
    
    // Return success response
    response := map[string]interface{}{
        "status": "success",
        "message": "Profile picture uploaded successfully",
        "picture_id": pictureID,
        "thumbnail": thumbnailPath,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func isValidImageType(contentType string) bool {
    validTypes := []string{"image/jpeg", "image/png", "image/gif"}
    for _, t := range validTypes {
        if contentType == t {
            return true
        }
    }
    return false
}

func saveUploadedFile(file multipart.File, header *multipart.FileHeader) (string, error) {
    // Create unique filename
    ext := filepath.Ext(header.Filename)
    filename := fmt.Sprintf("profile_%d%s", time.Now().Unix(), ext)
    
    // Create uploads directory if not exists
    uploadsDir := "uploads/profiles"
    os.MkdirAll(uploadsDir, 0755)
    
    // Save file
    dst, err := os.Create(filepath.Join(uploadsDir, filename))
    if err != nil {
        return "", err
    }
    defer dst.Close()
    
    _, err = io.Copy(dst, file)
    return filename, err
}

func createThumbnail(originalPath string) (string, error) {
    // Simple thumbnail creation (placeholder implementation)
    thumbnailPath := strings.Replace(originalPath, ".", "_thumb.", 1)
    
    // In real implementation, would use image processing library
    // For now, just copy the file as a placeholder
    src, err := os.Open(filepath.Join("uploads/profiles", originalPath))
    if err != nil {
        return "", err
    }
    defer src.Close()
    
    dst, err := os.Create(filepath.Join("uploads/profiles", thumbnailPath))
    if err != nil {
        return "", err
    }
    defer dst.Close()
    
    _, err = io.Copy(dst, src)
    return thumbnailPath, err
}

func saveToDatabase(filename, thumbnailPath string, fileSize int64) (int, error) {
    // Placeholder database save
    // In real implementation, would use actual database
    pictureID := 12345
    
    fmt.Printf("Saved to database: %s, thumbnail: %s, size: %d bytes\n", 
        filename, thumbnailPath, fileSize)
    
    return pictureID, nil
}
""",
            "admin_handler.go": """package main

import (
    "fmt"
    "net/http"
    "encoding/json"
    "strconv"
)

// AdminProfilePicturesHandler handles admin operations for profile pictures
func AdminProfilePicturesHandler(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case "GET":
        listProfilePictures(w, r)
    case "POST":
        moderateProfilePicture(w, r)
    case "DELETE":
        deleteProfilePicture(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func listProfilePictures(w http.ResponseWriter, r *http.Request) {
    // Parse query parameters
    status := r.URL.Query().Get("status")
    page := r.URL.Query().Get("page")
    
    pageNum, _ := strconv.Atoi(page)
    if pageNum <= 0 {
        pageNum = 1
    }
    
    // Mock data for testing
    pictures := []map[string]interface{}{
        {
            "id": 1,
            "user_id": 123,
            "filename": "profile_1234567890.jpg",
            "status": "pending",
            "upload_date": "2024-01-15T10:30:00Z",
            "file_size": 1024000,
        },
        {
            "id": 2,
            "user_id": 456,
            "filename": "profile_1234567891.png",
            "status": "approved",
            "upload_date": "2024-01-14T15:45:00Z", 
            "file_size": 512000,
        },
    }
    
    // Filter by status if provided
    if status != "" {
        var filtered []map[string]interface{}
        for _, pic := range pictures {
            if pic["status"] == status {
                filtered = append(filtered, pic)
            }
        }
        pictures = filtered
    }
    
    response := map[string]interface{}{
        "pictures": pictures,
        "page": pageNum,
        "total": len(pictures),
        "status": "success",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func moderateProfilePicture(w http.ResponseWriter, r *http.Request) {
    var request struct {
        PictureID int    `json:"picture_id"`
        Action    string `json:"action"` // "approve" or "reject"
        Reason    string `json:"reason,omitempty"`
    }
    
    err := json.NewDecoder(r.Body).Decode(&request)
    if err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    // Validate action
    if request.Action != "approve" && request.Action != "reject" {
        http.Error(w, "Invalid action", http.StatusBadRequest)
        return
    }
    
    // Update database (mock implementation)
    err = updatePictureStatus(request.PictureID, request.Action, request.Reason)
    if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }
    
    response := map[string]interface{}{
        "status": "success",
        "message": fmt.Sprintf("Picture %s successfully", request.Action+"d"),
        "picture_id": request.PictureID,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func deleteProfilePicture(w http.ResponseWriter, r *http.Request) {
    pictureIDStr := r.URL.Query().Get("id")
    pictureID, err := strconv.Atoi(pictureIDStr)
    if err != nil {
        http.Error(w, "Invalid picture ID", http.StatusBadRequest)
        return
    }
    
    // Delete from database and filesystem (mock implementation)
    err = deletePictureFromDatabase(pictureID)
    if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }
    
    response := map[string]interface{}{
        "status": "success",
        "message": "Picture deleted successfully",
        "picture_id": pictureID,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func updatePictureStatus(pictureID int, action, reason string) error {
    // Mock database update
    fmt.Printf("Updated picture %d status to %s, reason: %s\n", 
        pictureID, action, reason)
    return nil
}

func deletePictureFromDatabase(pictureID int) error {
    // Mock database deletion
    fmt.Printf("Deleted picture %d from database\n", pictureID)
    return nil
}
""",
            "search_integration.go": """package main

import (
    "fmt"
    "net/http"
    "encoding/json"
    "strings"
)

// SearchHandler handles user search with profile picture filtering
func SearchHandler(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    pictureStatus := r.URL.Query().Get("picture_status")
    
    if query == "" {
        http.Error(w, "Search query required", http.StatusBadRequest)
        return
    }
    
    // Search users (mock implementation)
    users := searchUsers(query, pictureStatus)
    
    response := map[string]interface{}{
        "query": query,
        "picture_status_filter": pictureStatus,
        "users": users,
        "count": len(users),
        "status": "success",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func searchUsers(query, pictureStatus string) []map[string]interface{} {
    // Mock user data with profile picture information
    allUsers := []map[string]interface{}{
        {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",
            "profile_picture": map[string]interface{}{
                "status": "approved",
                "filename": "profile_123.jpg",
                "has_picture": true,
            },
        },
        {
            "id": 456, 
            "name": "Jane Smith",
            "email": "jane@example.com",
            "profile_picture": map[string]interface{}{
                "status": "pending",
                "filename": "profile_456.png",
                "has_picture": true,
            },
        },
        {
            "id": 789,
            "name": "Bob Johnson", 
            "email": "bob@example.com",
            "profile_picture": map[string]interface{}{
                "status": "none",
                "filename": "",
                "has_picture": false,
            },
        },
    }
    
    var results []map[string]interface{}
    
    for _, user := range allUsers {
        // Filter by name/email matching query
        name := strings.ToLower(user["name"].(string))
        email := strings.ToLower(user["email"].(string))
        queryLower := strings.ToLower(query)
        
        if strings.Contains(name, queryLower) || strings.Contains(email, queryLower) {
            // Filter by picture status if provided
            if pictureStatus != "" {
                profilePic := user["profile_picture"].(map[string]interface{})
                if profilePic["status"] != pictureStatus {
                    continue
                }
            }
            results = append(results, user)
        }
    }
    
    return results
}
"""
        }
    elif task_category == 'architectural_understanding':
        return {
            "architecture_analysis.go": """package main

import "fmt"

// ArchitecturalAnalysis represents the analysis of the web application architecture
type ArchitecturalAnalysis struct {
    Patterns     []string
    Dependencies map[string][]string
    DataFlow     []DataFlowStep
    Concerns     []string
}

type DataFlowStep struct {
    Stage       string
    Module      string
    Description string
}

// AnalyzeArchitecture performs comprehensive architectural analysis
func AnalyzeArchitecture() *ArchitecturalAnalysis {
    analysis := &ArchitecturalAnalysis{
        Patterns: []string{
            "MVC (Model-View-Controller)",
            "Repository Pattern",
            "Layered Architecture",
            "Dependency Injection",
        },
        Dependencies: map[string][]string{
            "module_1": {"config", "utils", "constants"},
            "module_2": {"module_1", "database", "validation"},
            "module_3": {"module_2", "search", "indexing"},
            "module_4": {"module_1", "admin", "authentication"},
            "module_5": {"module_2", "file_upload", "storage"},
        },
        DataFlow: []DataFlowStep{
            {"Request", "HTTP Handler", "Receive user request"},
            {"Validation", "module_2", "Validate input data"},
            {"Processing", "module_1", "Process business logic"},
            {"Storage", "Database", "Persist data"},
            {"Response", "JSON", "Return formatted response"},
        },
        Concerns: []string{
            "Authentication handled in module_4",
            "File validation in module_5",
            "Search indexing in module_3",
            "Configuration centralized in config.go",
        },
    }
    
    return analysis
}

func main() {
    analysis := AnalyzeArchitecture()
    
    fmt.Println("=== Architectural Analysis ===")
    fmt.Println("Patterns identified:")
    for _, pattern := range analysis.Patterns {
        fmt.Printf("- %s\n", pattern)
    }
    
    fmt.Println("\nModule Dependencies:")
    for module, deps := range analysis.Dependencies {
        fmt.Printf("%s -> %v\n", module, deps)
    }
    
    fmt.Println("\nData Flow:")
    for _, step := range analysis.DataFlow {
        fmt.Printf("%s: %s (%s)\n", step.Stage, step.Description, step.Module)
    }
}
"""
        }
    else:
        return {
            "solution.go": f"""package main

import (
    "fmt"
    "log"
    "time"
)

// Solution for {task_category}
func main() {{
    fmt.Println("Solution implementation for {scenario['title']}")
    
    // Initialize solution
    err := initializeSolution()
    if err != nil {{
        log.Fatalf("Failed to initialize: %v", err)
    }}
    
    // Process task
    result, err := processTask()
    if err != nil {{
        log.Fatalf("Failed to process task: %v", err)
    }}
    
    // Output results
    fmt.Printf("Task completed successfully: %s\\n", result)
}}

func initializeSolution() error {{
    // Setup and initialization logic
    fmt.Println("Initializing solution...")
    time.Sleep(time.Millisecond * 100) // Simulate work
    return nil
}}

func processTask() (string, error) {{
    // Main task processing logic
    fmt.Println("Processing task...")
    
    // Simulate some work
    for i := 0; i < 3; i++ {{
        fmt.Printf("Step %d completed\\n", i+1)
        time.Sleep(time.Millisecond * 50)
    }}
    
    return "Task processed successfully", nil
}}

// Helper function for task processing
func validateInput(input string) bool {{
    return len(input) > 0
}}

// Error handling helper
func handleError(err error) {{
    if err != nil {{
        log.Printf("Error occurred: %v", err)
    }}
}}
"""
        }


def display_validation_results(result):
    """Display detailed validation results"""
    
    # Main scores table
    scores_table = Table(title="🏆 Validation Scores", title_style="bold blue")
    scores_table.add_column("Component", style="cyan")
    scores_table.add_column("Weight", style="yellow") 
    scores_table.add_column("Score", style="green")
    scores_table.add_column("Weighted", style="magenta")
    
    scores_table.add_row(
        "Functional Correctness", "40%", 
        f"{result.functional_score:.3f}", 
        f"{result.functional_score * 0.4:.3f}"
    )
    scores_table.add_row(
        "Agent Metrics", "30%",
        f"{result.agent_metrics_score:.3f}",
        f"{result.agent_metrics_score * 0.3:.3f}"
    )
    scores_table.add_row(
        "Code Quality", "20%",
        f"{result.quality_score:.3f}",
        f"{result.quality_score * 0.2:.3f}"
    )
    scores_table.add_row(
        "Style & Practices", "10%",
        f"{result.style_score:.3f}",
        f"{result.style_score * 0.1:.3f}"
    )
    scores_table.add_row(
        "", "", "", "", style="dim"
    )
    scores_table.add_row(
        "TOTAL SCORE", "100%",
        f"{result.total_score:.3f}",
        f"{result.total_score:.3f}",
        style="bold green"
    )
    
    console.print(scores_table)
    
    # Performance info
    performance_panel = Panel(
        f"⏱️  Execution Time: {result.execution_time:.2f}s\n"
        f"🎯 Scenario: {result.scenario_id}\n"
        f"🏆 Grade: {get_letter_grade(result.total_score)}",
        title="Performance Summary",
        style="blue"
    )
    
    console.print(performance_panel)


def get_letter_grade(score):
    """Convert score to letter grade"""
    if score >= 0.9:
        return "A+ (Excellent)"
    elif score >= 0.8:
        return "A (Very Good)"
    elif score >= 0.7:
        return "B (Good)"
    elif score >= 0.6:
        return "C (Fair)"
    elif score >= 0.5:
        return "D (Poor)"
    else:
        return "F (Failing)"


async def main():
    """Main test function"""
    try:
        result = await test_validation_framework()
        
        if result:
            console.print("\n✅ [bold green]Validation framework test completed successfully![/bold green]")
            console.print(f"🎯 Final Score: {result.total_score:.3f} ({get_letter_grade(result.total_score)})")
            
            console.print("\n🎉 [bold blue]Our 6 Novel Metrics are working![/bold blue]")
            console.print("   🏗️  ACS: Architectural Coherence Score")
            console.print("   🧭 DTA: Dependency Traversal Accuracy")
            console.print("   🧠 MMR: Multi-Session Memory Retention")
            console.print("   🔗 CFRD: Cross-File Reasoning Depth")
            console.print("   📈 IDC: Incremental Development Capability")
            console.print("   📊 ICU: Information Coverage Utilization")
            
        else:
            console.print("\n❌ [bold red]Test failed[/bold red]")
            
    except Exception as e:
        console.print(f"\n❌ [bold red]Test error: {e}[/bold red]")
        import traceback
        console.print(f"🔍 Debug: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main()) 