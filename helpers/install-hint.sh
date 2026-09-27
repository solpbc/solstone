# BEGIN shared install_hint
# One copy lives in solstone helpers/install-hint.sh (embedded into install.sh),
# the other in solstone-journal core/distribution/install.sh. Keep the two
# byte-identical: both repositories pin the same SHA-256 of this block.
install_hint() {
	_hint_pkg=$1
	_hint_file=${2:-/etc/os-release}
	_hint_id=
	_hint_like=
	if [ -r "$_hint_file" ]; then
		_hint_id=$(awk -F= '$1 == "ID" {gsub(/^"|"$/, "", $2); print $2; exit}' "$_hint_file")
		_hint_like=$(awk -F= '$1 == "ID_LIKE" {gsub(/^"|"$/, "", $2); print $2; exit}' "$_hint_file")
	fi
	case " ${_hint_id} " in
	" fedora ") _hint_family=fedora ;;
	" rhel ") _hint_family=rhel ;;
	*)
		case " ${_hint_id} ${_hint_like} " in
		*" debian "* | *" ubuntu "*) _hint_family=debian ;;
		*" rhel "* | *" centos "*) _hint_family=el ;;
		*" fedora "*) _hint_family=fedora ;;
		*" arch "*) _hint_family=arch ;;
		*" suse "* | *" opensuse "*) _hint_family=suse ;;
		*) _hint_family= ;;
		esac
		;;
	esac
	case "${_hint_family}:${_hint_pkg}" in
	debian:*) printf 'sudo apt install %s' "$_hint_pkg" ;;
	fedora:*) printf 'sudo dnf install %s' "$_hint_pkg" ;;
	el:minisign) printf '%s' "sudo dnf install epel-release && sudo dnf install minisign" ;;
	rhel:minisign) printf '%s' "enable EPEL (https://docs.fedoraproject.org/en-US/epel/) and run sudo dnf install minisign" ;;
	el:* | rhel:*) printf 'sudo dnf install %s' "$_hint_pkg" ;;
	arch:*) printf 'sudo pacman -S %s' "$_hint_pkg" ;;
	suse:*) printf 'sudo zypper install %s' "$_hint_pkg" ;;
	*) printf 'install %s from your distribution'"'"'s packages' "$_hint_pkg" ;;
	esac
}
# END shared install_hint
