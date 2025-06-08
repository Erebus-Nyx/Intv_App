#!/bin/bash
# INTV Quick Installation Script
# This script downloads and runs the Python-based installation script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the INTV directory with install.py
if [ -f "install.py" ]; then
    print_info "Found install.py in current directory"
    INSTALL_SCRIPT="./install.py"
elif [ -f "/home/nyx/intv/install.py" ]; then
    print_info "Using install.py from /home/nyx/intv/"
    INSTALL_SCRIPT="/home/nyx/intv/install.py"
else
    print_error "install.py not found!"
    print_info "Please run this script from the INTV project directory"
    exit 1
fi

print_header "INTV Quick Installation"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    print_info "Please install Python 3.10 or later and try again"
    exit 1
fi

# Parse command line arguments
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            ARGS="$ARGS --dry-run"
            print_info "Dry run mode enabled"
            shift
            ;;
        --cpu-only)
            ARGS="$ARGS --cpu-only"
            print_info "CPU-only installation mode"
            shift
            ;;
        --gpu-only)
            ARGS="$ARGS --gpu-only"
            print_info "GPU-only installation mode"
            shift
            ;;
        --force-reinstall)
            ARGS="$ARGS --force-reinstall"
            print_info "Force reinstall mode enabled"
            shift
            ;;
        --skip-native)
            ARGS="$ARGS --skip-native"
            print_info "Skipping native dependency installation"
            shift
            ;;
        --skip-verification)
            ARGS="$ARGS --skip-verification"
            print_info "Skipping installation verification"
            shift
            ;;
        --help|-h)
            echo "INTV Quick Installation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run             Show what would be done without doing it"
            echo "  --cpu-only            Force CPU-only installation"
            echo "  --gpu-only            Only install if GPU is detected"
            echo "  --force-reinstall     Force reinstallation"
            echo "  --skip-native         Skip native dependency installation"
            echo "  --skip-verification   Skip installation verification"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Standard installation"
            echo "  $0 --dry-run          # See what would be installed"
            echo "  $0 --cpu-only         # CPU-only installation"
            echo "  $0 --force-reinstall  # Reinstall even if already installed"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check system requirements
print_info "Checking system requirements..."

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_NUM=$(echo $PYTHON_VERSION | sed 's/\.//')

if [ "$PYTHON_VERSION_NUM" -lt "310" ]; then
    print_error "Python 3.10 or later is required (found: $PYTHON_VERSION)"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This may cause permission issues with pipx."
    print_info "Consider running as a regular user for better isolation."
fi

# Check available disk space (rough estimate)
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 5000000 ]; then  # 5GB in KB
    print_warning "Low disk space detected. Installation requires ~5GB for models and dependencies."
fi

# Check internet connectivity
if ! ping -c 1 google.com &> /dev/null; then
    print_warning "Internet connectivity check failed. Installation requires internet access."
fi

# Run the Python installation script
print_info "Starting Python installation script..."
echo ""

if python3 "$INSTALL_SCRIPT" $ARGS; then
    print_success "Installation completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Restart your terminal or run: source ~/.bashrc"
    echo "  2. Test the installation: intv --version"
    echo "  3. Check platform info: intv-platform"
    echo "  4. Get help: intv --help"
    echo ""
    print_info "Example usage:"
    echo "  intv process document.pdf"
    echo "  intv module create --interactive"
    echo "  intv module list"
else
    EXIT_CODE=$?
    print_error "Installation failed with exit code $EXIT_CODE"
    echo ""
    print_info "Troubleshooting:"
    echo "  1. Try running with --dry-run to see what would be installed"
    echo "  2. Check system requirements and dependencies"
    echo "  3. Run with elevated permissions if needed (sudo)"
    echo "  4. Check internet connectivity"
    echo ""
    exit $EXIT_CODE
fi
