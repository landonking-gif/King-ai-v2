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
import difflib
from pathlib import Path
from datetime import datetime

class RalphLoop:
    # Configuration constants
    FUZZY_MATCH_THRESHOLD = 0.85  # 85% similarity threshold for fuzzy matching
    MAX_CONTEXT_FILES = 5  # Maximum files to include in AI context
    MAX_FILE_SIZE_FOR_CONTEXT = 50000  # Skip files larger than 50KB for context
    
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
            with open(self.progress_file, 'w') as f:
                f.write("# Ralph Progress Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Project: {self.project_root.name}\n\n")

    def _get_branch_name(self):
        """Extract branch name from PRD"""
        with open(self.prd_file, 'r') as f:
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
        with open(self.prd_file, 'r') as f:
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
        with open(self.prompt_file, 'r') as f:
            template = f.read()

        # Load full PRD for context
        with open(self.prd_file, 'r') as f:
            prd_data = json.load(f)

        # Replace placeholders
        prompt = template.replace('{{STORY_TITLE}}', title)
        prompt = prompt.replace('{{STORY_DESCRIPTION}}', description)
        prompt = prompt.replace('{{STORY_ACCEPTANCE}}', acceptance)
        prompt = prompt.replace('{{ITERATION}}', str(iteration))

        # Add full PRD context
        prd_context = json.dumps(prd_data, indent=2)
        prompt = prompt.replace('{{PRD_CONTEXT}}', prd_context)

        # Add progress context
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress_context = f.read().split('\n')[-20:]  # Last 20 lines
            progress_context = '\n'.join(progress_context)
        else:
            progress_context = "No progress yet - this is the first iteration"
        
        prompt = prompt.replace('{{PROGRESS_CONTEXT}}', progress_context)

        return prompt

    async def _generate_code(self, prompt):
        """Generate code using GitHub Copilot CLI with enhanced file reading"""
        try:
            print("🤖 Generating code using GitHub Copilot CLI...")
            
            # Enhance prompt with current file contents if needed
            enhanced_prompt = await self._enhance_prompt_with_context(prompt)
            
            # Save prompt to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(enhanced_prompt)
                prompt_file = f.name
            
            try:
                # Try using 'gh copilot suggest' command for shell/code generation
                result = subprocess.run(
                    ['gh', 'copilot', 'suggest', '-t', 'shell', enhanced_prompt[:500]],  # Shortened for CLI
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0 and result.stdout:
                    print(f"✅ Generated code from Copilot")
                    return result.stdout
                else:
                    # Try Python script approach for better code generation
                    print("Using Python-based code generation...")
                    code_result = await self._generate_with_python(enhanced_prompt)
                    if code_result:
                        return code_result
                    
                    # Final fallback: try calling copilot directly with stdin
                    print("Trying alternative copilot invocation...")
                    with open(prompt_file, 'r', encoding='utf-8') as pf:
                        result = subprocess.run(
                            ['copilot'],
                            stdin=pf,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                    
                    if result.returncode == 0 and result.stdout:
                        print(f"✅ Generated code from Copilot")
                        return result.stdout
                    else:
                        print(f"❌ Copilot CLI failed: {result.stderr}")
                        print("")
                        print("Make sure GitHub Copilot CLI is installed and authenticated:")
                        print("  1. Install: gh extension install github/gh-copilot")
                        print("  2. Authenticate: gh auth login")
                        print("  3. Test: gh copilot --version")
                        return None
                        
            finally:
                # Clean up temp file
                try:
                    Path(prompt_file).unlink()
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            print("❌ Copilot CLI timed out after 5 minutes")
            return None
        except FileNotFoundError:
            print("❌ GitHub Copilot CLI not found")
            print("")
            print("Please install GitHub Copilot CLI:")
            print("  gh extension install github/gh-copilot")
            print("  gh auth login")
            return None
        except Exception as e:
            print(f"❌ AI code generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _enhance_prompt_with_context(self, prompt):
        """Enhance prompt with current file contents for better editing"""
        # Extract file paths mentioned in the prompt
        import re
        file_pattern = r'(?:src/|scripts/|config/|[\w\-]+/)[\w\-/]+\.(?:py|js|ts|tsx|json|yaml|yml|md)'
        mentioned_files = re.findall(file_pattern, prompt)
        
        enhanced = prompt
        if mentioned_files:
            enhanced += "\n\n## Current File Contents for Context\n"
            for file_path in mentioned_files[:self.MAX_CONTEXT_FILES]:  # Use class constant
                full_path = self.project_root / file_path
                if full_path.exists() and full_path.stat().st_size < self.MAX_FILE_SIZE_FOR_CONTEXT:
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        enhanced += f"\n### {file_path}\n```\n{content}\n```\n"
                    except Exception:
                        pass  # Skip files that can't be read
        
        return enhanced
    
    async def _generate_with_python(self, prompt):
        """Use Python-based generation script as fallback"""
        try:
            script_path = self.project_root / 'scripts' / 'ralph' / 'generate_code.py'
            if not script_path.exists():
                return None
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=prompt.encode())
            
            if process.returncode == 0 and stdout:
                print("✅ Generated code using Python script")
                return stdout.decode().strip()
            
            return None
        except Exception as e:
            print(f"Python generation fallback failed: {e}")
            return None

    def _apply_code_changes(self, generated_code):
        """Apply the generated code changes to files with enhanced editing capabilities"""
        import re

        changes_made = 0

        # Pattern 1: filepath blocks - ```filepath: path/to/file.ext
        filepath_pattern = r'```filepath:\s*([^\n]+)\n(.*?)\n```'
        filepath_matches = re.findall(filepath_pattern, generated_code, re.MULTILINE | re.DOTALL)

        for file_path, code in filepath_matches:
            file_path = file_path.strip()
            
            # Make path relative to project root
            full_path = self.project_root / file_path
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the code
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code.strip() + '\n')

            print(f"📝 Created/updated: {file_path}")
            changes_made += 1

        # Pattern 2: edit blocks - ```edit: path/to/file.ext with SEARCH/REPLACE
        edit_pattern = r'```edit:\s*([^\n]+)\nSEARCH:\n(.*?)\n\nREPLACE:\n(.*?)\n```'
        edit_matches = re.findall(edit_pattern, generated_code, re.MULTILINE | re.DOTALL)

        for file_path, search_text, replace_text in edit_matches:
            file_path = file_path.strip()
            full_path = self.project_root / file_path

            if not full_path.exists():
                print(f"⚠️  File not found for edit: {file_path}")
                continue

            # Read existing content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply the replacement
            search_text = search_text.strip()
            replace_text = replace_text.strip()
            
            if search_text in content:
                # Count occurrences to warn about multiple matches
                occurrence_count = content.count(search_text)
                if occurrence_count > 1:
                    print(f"⚠️  Warning: Found {occurrence_count} occurrences, replacing first only")
                
                new_content = content.replace(search_text, replace_text, 1)  # Replace first occurrence only
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✏️  Edited: {file_path}")
                changes_made += 1
            else:
                # Try fuzzy matching for minor whitespace differences
                lines = content.split('\n')
                search_lines = search_text.split('\n')
                
                # Try to find approximate match
                for i in range(len(lines) - len(search_lines) + 1):
                    block = '\n'.join(lines[i:i+len(search_lines)])
                    similarity = difflib.SequenceMatcher(None, search_text, block).ratio()
                    
                    if similarity > self.FUZZY_MATCH_THRESHOLD:
                        print(f"⚠️  Found approximate match (similarity: {similarity:.1%}) in {file_path}")
                        new_lines = lines[:i] + replace_text.split('\n') + lines[i+len(search_lines):]
                        new_content = '\n'.join(new_lines)
                        
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        print(f"✏️  Edited with fuzzy match: {file_path}")
                        changes_made += 1
                        break
                else:
                    print(f"⚠️  Search text not found in {file_path}")
                    print(f"    First few lines of search: {search_text[:100]}...")

        # Pattern 3: Insert/append blocks - ```insert: path/to/file.ext after/before LINE
        insert_pattern = r'```insert:\s*([^\n]+)\s+(after|before)\s+"([^"]+)"\n(.*?)\n```'
        insert_matches = re.findall(insert_pattern, generated_code, re.MULTILINE | re.DOTALL)
        
        for file_path, position, marker, code_to_insert in insert_matches:
            file_path = file_path.strip()
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                print(f"⚠️  File not found for insert: {file_path}")
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if marker in content:
                if position == 'after':
                    new_content = content.replace(marker, marker + '\n' + code_to_insert.strip())
                else:  # before
                    new_content = content.replace(marker, code_to_insert.strip() + '\n' + marker)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"📌 Inserted code {position} marker in: {file_path}")
                changes_made += 1
            else:
                print(f"⚠️  Marker '{marker}' not found in {file_path}")

        # Pattern 4: standard code blocks (fallback) - ```language
        if changes_made == 0:
            # Try standard markdown code blocks
            standard_pattern = r'```(?:[\w]+)?\s*([^\n]*)\n(.*?)\n```'
            standard_matches = re.findall(standard_pattern, generated_code, re.MULTILINE | re.DOTALL)

            for header, code in standard_matches:
                # Skip if this looks like it was already processed
                if header.startswith('filepath:') or header.startswith('edit:') or header.startswith('insert:'):
                    continue
                
                # Try to extract filename from header or context
                if header and not header.isalpha():
                    file_path = header.strip()
                    
                    full_path = self.project_root / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(code.strip() + '\n')

                    print(f"📝 Created/updated: {file_path}")
                    changes_made += 1

        if changes_made == 0:
            print("⚠️  No file changes detected in AI output")
            print("The AI may have provided explanatory text without code blocks.")
            print("Please check the output manually.")
        else:
            print(f"✅ Applied {changes_made} file change(s)")

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
    
    def _create_or_update_pr(self, branch_name):
        """Create or update a pull request using GitHub CLI"""
        try:
            # Check if PR already exists
            result = subprocess.run(
                ['gh', 'pr', 'list', '--head', branch_name, '--json', 'number'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != '[]':
                pr_data = json.loads(result.stdout)
                if pr_data:
                    pr_number = pr_data[0]['number']
                    print(f"✅ PR #{pr_number} already exists for branch {branch_name}")
                    return pr_number
            
            # Create new PR
            with open(self.prd_file, 'r') as f:
                prd_data = json.load(f)
            
            # Generate PR body from completed stories
            completed_stories = [s for s in prd_data['userStories'] if s['passes']]
            pr_body = "## Completed User Stories\n\n"
            for story in completed_stories:
                pr_body += f"- [x] **{story['title']}** ({story['id']})\n"
                pr_body += f"  - {story['description']}\n\n"
            
            incomplete_stories = [s for s in prd_data['userStories'] if not s['passes']]
            if incomplete_stories:
                pr_body += "\n## Remaining Stories\n\n"
                for story in incomplete_stories:
                    pr_body += f"- [ ] **{story['title']}** ({story['id']})\n"
            
            # Create PR
            result = subprocess.run(
                ['gh', 'pr', 'create', 
                 '--title', f"Ralph: Implementing {branch_name}",
                 '--body', pr_body,
                 '--base', 'main'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Extract PR number from output
                import re
                match = re.search(r'#(\d+)', result.stdout)
                if match:
                    pr_number = match.group(1)
                    print(f"✅ Created PR #{pr_number}")
                    return int(pr_number)
                print(f"✅ Created PR: {result.stdout.strip()}")
                return None
            else:
                print(f"⚠️  Could not create PR: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"⚠️  PR creation error: {e}")
            return None
    
    def _comment_on_pr(self, pr_number, comment):
        """Add a comment to a PR using GitHub CLI"""
        try:
            result = subprocess.run(
                ['gh', 'pr', 'comment', str(pr_number), '--body', comment],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"✅ Added comment to PR #{pr_number}")
                return True
            else:
                print(f"⚠️  Could not comment on PR: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠️  PR comment error: {e}")
            return False

    def _mark_story_complete(self, story_id):
        """Mark a story as completed in the PRD"""
        with open(self.prd_file, 'r') as f:
            data = json.load(f)

        for story in data['userStories']:
            if story['id'] == story_id:
                story['passes'] = True
                break

        with open(self.prd_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _mark_story_failed(self, story_id):
        """Mark a story as failed in the PRD (still incomplete but flagged)"""
        with open(self.prd_file, 'r') as f:
            data = json.load(f)

        for story in data['userStories']:
            if story['id'] == story_id:
                story['passes'] = False
                if 'metadata' not in story:
                    story['metadata'] = {}
                story['metadata']['failed'] = True
                story['metadata']['failedAt'] = datetime.now().isoformat()
                break

        with open(self.prd_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _commit_changes(self, story_id, story_title):
        """Commit the changes to git and optionally push/create PR"""
        try:
            subprocess.run(['git', 'add', '.'], check=True)
            commit_msg = f"Ralph: Complete story {story_id} - {story_title}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print(f"Committed changes: {commit_msg}")
            
            # Try to push to remote (optional - won't fail if push fails)
            try:
                branch_name = self._get_branch_name()
                result = subprocess.run(['git', 'push', '-u', 'origin', branch_name], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ Pushed to remote branch: {branch_name}")
                    
                    # Try to create or update PR
                    pr_number = self._create_or_update_pr(branch_name)
                    if pr_number:
                        # Add comment about completed story
                        comment = f"✅ Completed story `{story_id}`: **{story_title}**\n\nCommit: {commit_msg}"
                        self._comment_on_pr(pr_number, comment)
                else:
                    print(f"⚠️  Could not push to remote: {result.stderr}")
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"⚠️  Push/PR creation skipped: {e}")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git commit failed: {e}")

    def _update_progress(self, iteration, story_id, story_title, success=True, error=None):
        """Update the progress log"""
        with open(self.progress_file, 'a') as f:
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
        with open('prd.json', 'r') as f:
            data = json.load(f)

        for story in data['userStories']:
            story['passes'] = False
            # Remove failed metadata
            if 'metadata' in story:
                story['metadata'].pop('failed', None)
                story['metadata'].pop('failedAt', None)

        with open('prd.json', 'w') as f:
            json.dump(data, f, indent=2)

        print("All stories reset")
        return

    # Run the Ralph loop
    loop = RalphLoop(max_iterations=args.max_iterations, max_retries_per_story=args.max_retries)

    try:
        if sys.platform == 'win32':
            # Use proper event loop policy for Windows to avoid cleanup issues
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
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