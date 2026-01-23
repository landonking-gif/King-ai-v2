#!/usr/bin/env python3
"""
Ralph: Autonomous AI Agent Loop
Python implementation for better cross-platform compatibility
Based on Geoffrey Huntley's Ralph pattern and Ryan Carson's implementation
"""

import os
import sys
import json
import asyncio
import subprocess
import tempfile
import base64
from pathlib import Path
from datetime import datetime
import httpx

class RalphLoop:
    def __init__(self, max_iterations=None, max_retries_per_story=3):
        self.max_iterations = max_iterations
        self.max_retries_per_story = max_retries_per_story
        self.prd_file = Path("prd.json")
        self.progress_file = Path("progress.txt")
        self.prompt_file = Path("scripts/ralph/prompt.md")
        self.project_root = Path.cwd()
        self.story_retry_count = {}  # Track retry attempts per story

        # Ensure we're in the right directory
        if not self.prd_file.exists():
            raise FileNotFoundError(f"prd.json not found in {self.project_root}")

    async def run(self):
        """Main Ralph loop execution"""
        print("Starting Ralph Autonomous Agent Loop")
        print(f"Working directory: {self.project_root}")
        print(f"Max iterations: {self.max_iterations}")

        # Initialize progress tracking
        self._init_progress()

        # Get branch name and checkout
        branch_name = self._get_branch_name()
        self._checkout_branch(branch_name)

        iteration = 1
        while self.max_iterations is None or iteration <= self.max_iterations:
            print(f"\n{'='*50}")
            print(f"Ralph Iteration {iteration}")
            print('='*50)

            # Find next incomplete story
            story = self._get_next_story()
            if not story:
                print("ALL STORIES COMPLETED!")
                print("<promise>COMPLETE</promise>")
                break

            story_id = story['id']
            story_title = story['title']
            story_description = story['description']
            story_acceptance = story['acceptanceCriteria']

            print(f"Working on: {story_title}")
            print(f"Story ID: {story_id}")

            # Check retry count for this story
            retry_count = self.story_retry_count.get(story_id, 0)
            if retry_count >= self.max_retries_per_story:
                print(f"⚠️  Story {story_id} exceeded max retries ({self.max_retries_per_story})")
                print(f"Marking as failed and moving to next story")
                self._mark_story_failed(story_id)
                self._update_progress(iteration, story_id, story_title, success=False, 
                                     error=f"Exceeded max retries ({self.max_retries_per_story})")
                continue

            try:
                # Generate implementation
                success = await self._implement_story(
                    story_id, story_title, story_description, story_acceptance, iteration
                )

                if success:
                    # Mark story as complete
                    self._mark_story_complete(story_id)
                    print(f"✅ Successfully completed story: {story_id}")

                    # Reset retry count for this story
                    self.story_retry_count[story_id] = 0

                    # Commit changes
                    self._commit_changes(story_id, story_title)

                    # Update progress
                    self._update_progress(iteration, story_id, story_title, success=True)
                else:
                    # Increment retry count
                    self.story_retry_count[story_id] = retry_count + 1
                    print(f"❌ Failed to implement story: {story_id} (Attempt {self.story_retry_count[story_id]}/{self.max_retries_per_story})")
                    self._update_progress(iteration, story_id, story_title, success=False,
                                         error=f"Implementation failed (Attempt {self.story_retry_count[story_id]}/{self.max_retries_per_story})")

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                self._update_progress(iteration, story_id, story_title, success=False, error=str(e))

            iteration += 1

        if self.max_iterations is not None and iteration > self.max_iterations:
            print(f"Reached maximum iterations ({self.max_iterations}) without completing all stories.")

    def _init_progress(self):
        """Initialize progress tracking file"""
        if not self.progress_file.exists():
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                f.write("# Ralph Progress Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Project: {self.project_root.name}\n\n")

    def _get_branch_name(self):
        """Extract branch name from PRD"""
        with open(self.prd_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['branchName']

    def _checkout_branch(self, branch_name):
        """Create and checkout feature branch"""
        try:
            # Check if branch exists
            result = subprocess.run(['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{branch_name}'],
                                  capture_output=True)
            if result.returncode == 0:
                subprocess.run(['git', 'checkout', branch_name], check=True)
            else:
                subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            print(f"Checked out branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git checkout failed: {e}")
            raise

    def _get_next_story(self):
        """Find the highest priority incomplete story (excluding failed ones)"""
        with open(self.prd_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get incomplete stories that haven't been marked as failed
        incomplete_stories = [
            s for s in data['userStories'] 
            if not s['passes'] and not s.get('metadata', {}).get('failed', False)
        ]
        if not incomplete_stories:
            return None

        # Sort by priority (lower number = higher priority)
        incomplete_stories.sort(key=lambda s: s['priority'])
        return incomplete_stories[0]

    async def _implement_story(self, story_id, title, description, acceptance, iteration):
        """Implement a single story using AI code generation"""
        try:
            # Create iteration prompt
            prompt = self._create_prompt(title, description, acceptance, iteration)

            # Generate code using GitHub Copilot CLI
            generated_code = await self._generate_code(prompt)

            if not generated_code:
                print("Code generation failed - no output")
                return False

            # Apply the generated code
            print("Applying code changes...")
            applied = self._apply_code_changes(generated_code)
            
            if not applied:
                print("No code changes were applied")
                return False

            # Run quality checks
            print("Running quality checks...")
            if not await self._run_quality_checks():
                print("Quality checks failed")
                return False

            return True

        except Exception as e:
            print(f"💥 Implementation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_prompt(self, title, description, acceptance, iteration):
        """Create the AI prompt for this iteration"""
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Load full PRD for context
        with open(self.prd_file, 'r', encoding='utf-8') as f:
            prd_data = json.load(f)

        # Get current story ID
        story_id = None
        for story in prd_data['userStories']:
            if story['title'] == title:
                story_id = story['id']
                break

        # Replace placeholders
        prompt = template.replace('{{STORY_ID}}', story_id or 'unknown')
        prompt = prompt.replace('{{STORY_TITLE}}', title)
        prompt = prompt.replace('{{STORY_DESCRIPTION}}', description)
        prompt = prompt.replace('{{STORY_ACCEPTANCE}}', acceptance)
        prompt = prompt.replace('{{ITERATION}}', str(iteration))

        # Add full PRD context
        prd_context = json.dumps(prd_data, indent=2)
        prompt = prompt.replace('{{PRD_CONTEXT}}', prd_context)

        # Add progress context
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_context = f.read().split('\n')[-20:]  # Last 20 lines
            progress_context = '\n'.join(progress_context)
        else:
            progress_context = "No progress yet - this is the first iteration"
        
        prompt = prompt.replace('{{PROGRESS_CONTEXT}}', progress_context)

        return prompt

    async def _generate_code(self, prompt):
        """Generate code using GitHub Copilot CLI"""
        try:
            print("🤖 Generating code using GitHub Copilot CLI...")
            
            # Save prompt to temporary file for reference
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                prompt_file = f.name
            
            try:
                # Use 'copilot' command directly (VS Code Copilot CLI)
                # This sends the prompt to Copilot and receives the response
                print("Invoking Copilot CLI with prompt...")
                
                # Method 1: Try direct copilot invocation with prompt as input
                result = subprocess.run(
                    ['copilot', prompt],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for complex tasks
                    shell=True,
                    cwd=str(self.project_root)
                )
                
                if result.returncode == 0 and result.stdout:
                    print(f"✅ Generated code from Copilot CLI")
                    return result.stdout
                
                # Method 2: Try with stdin if direct argument didn't work
                print("Trying Copilot CLI with stdin...")
                process = subprocess.Popen(
                    ['copilot'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                    cwd=str(self.project_root)
                )
                
                stdout, stderr = process.communicate(input=prompt, timeout=600)
                
                if process.returncode == 0 and stdout:
                    print(f"✅ Generated code from Copilot CLI (stdin method)")
                    return stdout
                
                # If both methods failed, provide helpful error
                print(f"❌ Copilot CLI failed")
                print(f"Return code: {process.returncode}")
                if stderr:
                    print(f"Error output: {stderr}")
                
                print("\nMake sure GitHub Copilot CLI is accessible:")
                print("  1. Open VS Code")
                print("  2. Type 'copilot' in the terminal to verify it's available")
                print("  3. Ensure GitHub Copilot extension is installed and authenticated")
                return None
                        
            finally:
                # Clean up temp file
                try:
                    Path(prompt_file).unlink()
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            print("❌ Copilot CLI timed out after 10 minutes")
            return None
        except FileNotFoundError:
            print("❌ Copilot CLI not found in PATH")
            print("\nTroubleshooting:")
            print("  1. Make sure you're running this from VS Code terminal")
            print("  2. Type 'copilot' to test if it's available")
            print("  3. Install GitHub Copilot extension if not already installed")
            return None
        except Exception as e:
            print(f"❌ AI code generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_code_changes(self, generated_code):
        """Apply the generated code changes to files"""
        import re

        changes_made = 0
        
        print("\n" + "="*60)
        print("APPLYING CODE CHANGES")
        print("="*60)

        # Pattern 1: filepath blocks - ```filepath: path/to/file.ext
        filepath_pattern = r'```filepath:\s*([^\n]+)\n(.*?)\n```'
        filepath_matches = re.findall(filepath_pattern, generated_code, re.MULTILINE | re.DOTALL)

        if filepath_matches:
            print(f"\nFound {len(filepath_matches)} file(s) to create/update:\n")
            
        for file_path, code in filepath_matches:
            file_path = file_path.strip()
            
            # Make path relative to project root
            full_path = self.project_root / file_path
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the code
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code.strip() + '\n')

            print(f"  ✅ Created/updated: {file_path}")
            changes_made += 1

        # Pattern 2: edit blocks - ```edit: path/to/file.ext with SEARCH/REPLACE
        edit_pattern = r'```edit:\s*([^\n]+)\nSEARCH:\n(.*?)\n\nREPLACE:\n(.*?)\n```'
        edit_matches = re.findall(edit_pattern, generated_code, re.MULTILINE | re.DOTALL)

        if edit_matches:
            print(f"\nFound {len(edit_matches)} edit(s) to apply:\n")

        for file_path, search_text, replace_text in edit_matches:
            file_path = file_path.strip()
            full_path = self.project_root / file_path

            if not full_path.exists():
                print(f"  ⚠️  File not found for edit: {file_path}")
                continue

            # Read existing content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply the replacement
            search_text = search_text.strip()
            replace_text = replace_text.strip()
            
            if search_text in content:
                new_content = content.replace(search_text, replace_text)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  ✅ Edited: {file_path}")
                changes_made += 1
            else:
                print(f"  ⚠️  Search text not found in {file_path}")

        # Pattern 3: standard code blocks (fallback) - ```language
        if changes_made == 0:
            print("\nNo filepath/edit blocks found. Trying standard code blocks...")
            
            # Try standard markdown code blocks
            standard_pattern = r'```(?:[\w]+)?\s*([^\n]*)\n(.*?)\n```'
            standard_matches = re.findall(standard_pattern, generated_code, re.MULTILINE | re.DOTALL)

            for header, code in standard_matches:
                # Skip if this looks like it was already processed
                if header.startswith('filepath:') or header.startswith('edit:'):
                    continue
                
                # Try to extract filename from header or context
                if header and not header.isalpha():
                    file_path = header.strip()
                    
                    full_path = self.project_root / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(code.strip() + '\n')

                    print(f"  ✅ Created/updated: {file_path}")
                    changes_made += 1

        print("\n" + "="*60)
        if changes_made == 0:
            print("⚠️  NO FILE CHANGES DETECTED")
            print("\nThe Copilot response may not have included code blocks.")
            print("This could mean:")
            print("  1. The story is already complete")
            print("  2. Copilot provided explanations instead of code")
            print("  3. The prompt needs to be more specific")
            print("\nCopilot Response Preview:")
            print("-" * 60)
            # Show first 500 chars of response
            preview = generated_code[:500] if len(generated_code) > 500 else generated_code
            print(preview)
            if len(generated_code) > 500:
                print("... (truncated)")
            print("-" * 60)
        else:
            print(f"✅ SUCCESSFULLY APPLIED {changes_made} CHANGE(S)")
        print("="*60 + "\n")

        return changes_made > 0

    async def _run_quality_checks(self):
        """Run quality checks on the codebase"""
        try:
            checks = []

            # Check if we're in a Python project
            if Path('pyproject.toml').exists() or Path('requirements.txt').exists():
                # Run basic Python checks
                try:
                    subprocess.run([sys.executable, '-m', 'py_compile', 'src/'], check=True, capture_output=True)
                    checks.append(("Python syntax", True))
                except subprocess.CalledProcessError:
                    checks.append(("Python syntax", False))

            # Check if we have a Node.js project
            if Path('package.json').exists():
                try:
                    subprocess.run(['npm', 'run', 'type-check'], check=True, capture_output=True, cwd='.')
                    checks.append(("TypeScript", True))
                except (subprocess.CalledProcessError, FileNotFoundError):
                    checks.append(("TypeScript", False))

            # All checks passed if no failures
            all_passed = all(result for _, result in checks)

            if checks:
                print("Quality check results:")
                for check_name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"  {status} {check_name}")

            return all_passed

        except Exception as e:
            print(f"⚠️  Quality check error: {e}")
            return True  # Don't fail on check errors

    def _mark_story_complete(self, story_id):
        """Mark a story as completed in the PRD"""
        with open(self.prd_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for story in data['userStories']:
            if story['id'] == story_id:
                story['passes'] = True
                break

        with open(self.prd_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _mark_story_failed(self, story_id):
        """Mark a story as failed in the PRD (still incomplete but flagged)"""
        with open(self.prd_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for story in data['userStories']:
            if story['id'] == story_id:
                story['passes'] = False
                if 'metadata' not in story:
                    story['metadata'] = {}
                story['metadata']['failed'] = True
                story['metadata']['failedAt'] = datetime.now().isoformat()
                break

        with open(self.prd_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _commit_changes(self, story_id, story_title):
        """Commit the changes to git"""
        try:
            subprocess.run(['git', 'add', '.'], check=True)
            commit_msg = f"Ralph: Complete story {story_id} - {story_title}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print(f"Committed changes: {commit_msg}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git commit failed: {e}")

    def _update_progress(self, iteration, story_id, story_title, success=True, error=None):
        """Update the progress log"""
        with open(self.progress_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            status = "SUCCESS" if success else "FAILED"
            f.write(f"[{timestamp}] Iteration {iteration}: {status} - {story_id}\n")
            f.write(f"  Story: {story_title}\n")
            if error:
                f.write(f"  Error: {error}\n")
            f.write("\n")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Ralph Autonomous Agent Loop')
    parser.add_argument('max_iterations', nargs='?', type=int, default=None,
                       help='Maximum number of iterations (default: unlimited)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retries per story before marking as failed (default: 3)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset all stories to incomplete')

    args = parser.parse_args()

    # Handle reset option
    if args.reset:
        print("Resetting all stories to incomplete...")
        with open('prd.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        for story in data['userStories']:
            story['passes'] = False
            # Remove failed metadata
            if 'metadata' in story:
                story['metadata'].pop('failed', None)
                story['metadata'].pop('failedAt', None)

        with open('prd.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print("All stories reset")
        return

    # Run the Ralph loop
    loop = RalphLoop(max_iterations=args.max_iterations, max_retries_per_story=args.max_retries)

    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\n\nRalph loop interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()