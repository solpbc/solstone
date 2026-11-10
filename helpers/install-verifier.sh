# BEGIN shared install_verifier
# Keep this block in sync with solstone-journal core/distribution/install.sh.
# The pins cover the unmodified Linux executables from minisign 0.12.
select_minisign() {
	MINISIGN=$(command -v minisign 2>/dev/null) && [ -x "$MINISIGN" ] && return 0
	[ "$3" = online ] || return 1

	case "$1" in
	x86_64) _ms_sha=2c74dffcc1c9a5ee55957c60971998ace2b89f22585631594ec2152c588af8db ;;
	aarch64) _ms_sha=cec9f88be8c975af76854a53b4d49c3d257feae38d916edb0d16fb55aacd3000 ;;
	*) return 1 ;;
	esac
	_ms_dir="$2/verifier"
	_ms_file="$_ms_dir/minisign"
	_ms_url="https://solstone.app/install-tools/minisign/0.12/$1/minisign"
	mkdir -m 0700 "$_ms_dir" || return 1
	# Cap the write even if the server omits a content length.
	(
		ulimit -f 1024 || exit 1
		if command -v curl >/dev/null 2>&1; then
			_ms_status=$(curl -q --proto '=https' -fsS --connect-timeout 10 --max-time 60 \
				-o "$_ms_file" -w '%{http_code}' "$_ms_url") && [ "$_ms_status" = 200 ]
		else
			wget --no-config --https-only --max-redirect=0 --tries=2 --timeout=30 -q -O "$_ms_file" "$_ms_url"
		fi
	) || return 1
	_ms_actual=$(digest_file "$_ms_file") || return 1
	[ "$_ms_actual" = "$_ms_sha" ] || return 1
	chmod 0700 "$_ms_file" || return 1
	MINISIGN=$_ms_file
	# A catalogue can delegate to an older signed journal bootstrap.
	PATH="$_ms_dir:$PATH"
	export PATH
}
# END shared install_verifier
